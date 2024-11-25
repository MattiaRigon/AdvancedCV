from collections import defaultdict
import os
import json
from typing import List
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import time

from tqdm import tqdm

from nxtp_ours.src.decoding import OneShotDecoder
from nxtp_ours.src.encoding import construct_embd_inputs, construct_text_inputs

import torchvision.transforms as T

from nxtp_ours.src.evals.engine import encode_cap_to_objs
from nxtp_ours.src.evals.metrics import SemanticFScore
from nxtp_ours.src.functions import load_clip, load_llama
from nxtp_ours.src.models.classifier import LangClassifier
from nxtp_ours.src.utils import load_checkpoint, load_config, set_dtype


from matplotlib import pyplot as plt
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset personalizzato per caricare immagini, JSON e testo.
        
        Args:
            root_dir (str): Percorso alla directory del dataset.
            transform (callable, optional): Trasformazioni da applicare alle immagini.
        """
        self.root_dir = root_dir

        # Trova tutti i prefissi (numeri_sample) presenti nel dataset
        self.sample_ids = sorted(
            set(os.path.splitext(f)[0] for f in os.listdir(root_dir))
        )
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Costruisci i percorsi per immagine, json e testo
        img_path = os.path.join(self.root_dir, f"{sample_id}.jpg")
        json_path = os.path.join(self.root_dir, f"{sample_id}.json")
        txt_path = os.path.join(self.root_dir, f"{sample_id}.txt")
    

        # Carica il file JSON
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        # Carica il testo
        with open(txt_path, "r") as f:
            text = f.read().strip()
        
        return {"json": json_data, "text": text, "img_path": img_path}
    
def build_preprocess_sam(input_size):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    # NOTE: because we freeze CLIP, won't apply augmentations on images for now
    resized_crop = [
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
    ]

    return T.Compose([*resized_crop, *to_rgb])

def build_preprocess(input_size):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    # NOTE: because we freeze CLIP, won't apply augmentations on images for now
    resized_crop = [
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
    ]

    norm = [
        T.ToTensor(),
        T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
    return T.Compose([*resized_crop, *to_rgb, *norm])

def create_tensor_from_mask(mask, patch_size):
    """Create a 16x16 tensor for the given mask, dividing it into patches."""
    h, w = mask.shape
    tensor = torch.zeros((patch_size, patch_size))
    
    patch_h, patch_w = h // patch_size, w // patch_size
    for i in range(patch_size):
        for j in range(patch_size):
            patch = mask[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
            if np.any(patch):
                tensor[i, j] = 1

    return tensor
    
def compute_masks(mask_generator,image_path):
    try:
        image = Image.open(image_path)
        image = build_preprocess_sam(224)(image)    
        image = np.array(image.convert("RGB"))
        masks = mask_generator.generate(image)
        tensors = []
        patch_size = 16
        
        for idx, mask in enumerate(masks):
            # Get the segmentation and convert it to a tensor
            segmentation = mask['segmentation']
            tensor = create_tensor_from_mask(segmentation, patch_size)
            tensors.append(tensor)

        return tensors
    except Exception as e:
        print(f"Error: {e}")
        return []

def predict(model,mask_generator,tokenizer,img_path,args,device = "cuda",num_labels=10):

    # load image
    img = Image.open(img_path).convert("RGB")
    img = build_preprocess(args.input_size)(img)
    img = img.unsqueeze(0).to(device)

    ctx = torch.amp.autocast(device_type="cuda", dtype=args.ptdtype)

    # infer
    t1 = time.perf_counter()
    with ctx:
        # get image token embeddings

        h = model.encode_images(img)
        z = model.decode_images(h)

        # drop [CLS] token embedding
        embds_clss, embds_imgs = z[:, :1], z[:, 1:]
        bs, n_img_tokens = embds_imgs.shape[:2]

        # convert text to tokens
        caps = ["" for _ in range(bs)]  # means no reference labels in prompt
        (
            tokens_caps,
            tokens_objs,
            _,
            _,
            dummy_token_index_cap,
            dummy_token_index_obj,
        ) = construct_text_inputs(
            args, caps, tokenizer, offset=n_img_tokens, is_train=False
        )
        tokens_caps = tokens_caps.to(device)
        tokens_objs = tokens_objs.to(device)

        # convert tokens to embeddings
        Wte = model.language_decoder.tok_embeddings.weight
        embds_caps = Wte[tokens_caps]
        embds_objs = Wte[tokens_objs]

        _, input_embds_objs, input_tokens_objs = construct_embd_inputs(
            embds_imgs,
            embds_caps,
            embds_objs,
            dummy_token_index_cap,
            dummy_token_index_obj,
            tokens_caps,
            tokens_objs,
            tokenizer,
        )

        # shave padding tokens
        shave_ind = torch.where(tokens_objs == tokenizer.pad_id)[1][0]
        input_tokens = input_tokens_objs[:, : shave_ind + n_img_tokens]
        input_embds = input_embds_objs[:, : shave_ind + n_img_tokens]

        # init text decoder for sampling
        text_decoder = OneShotDecoder(k=num_labels)
        text_decoder.reset()

        # init output tokens and logprobs
        tokens = tokens_objs[:, :shave_ind]  # will be final output tokens
        sum_logprobs = torch.zeros(bs, device=device)

        # visualize attention maps
        cached_tensors = None

        # start sampling
        x = input_embds
        all_preds = {}
        prev = []
        masked_input = None

        masks = compute_masks(mask_generator,img_path)
        for mask in masks:
            masked_input = 1- mask
            x = input_embds
            logits = model.language_decoder.forward(
                x,
                start_pos=0,
                dummy_token_index=dummy_token_index_obj,
                offset=n_img_tokens,
                input_tokens=input_tokens,
                prefix_image_tok_embeds=args.prefix_image_tok_embeds,
                decouple_label_tok_embeds=args.decouple_label_tok_embeds,
                is_train=False,
                cached_tensors=cached_tensors,
                masked_input=masked_input,
            )
            total = torch.zeros((16, 16))

            # get the initial tokens after the first forward pass
            tokens = tokens_objs[:, :shave_ind]
            sum_logprobs = torch.zeros(bs, device=device)
            tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
            next_tokens = tokens[:, -1].unsqueeze(1)

            # continue sampling until all labels reach [SEP]
            while completed == False:
                if x.shape[0] != next_tokens.shape[0]:
                    assert next_tokens.shape[0] % x.shape[0] == 0
                    x = x.repeat_interleave(next_tokens.shape[0] // x.shape[0], dim=0)

                # here we don't use the kv-attention for computing attention
                # if needed, can be added in the future
                x = torch.cat(
                    [
                        x,
                        Wte[next_tokens],
                    ],
                    dim=1,
                )

                logits = model.language_decoder.forward(
                    x,
                    start_pos=0,
                    dummy_token_index=dummy_token_index_obj,
                    offset=n_img_tokens,
                    input_tokens=input_tokens,
                    prefix_image_tok_embeds=args.prefix_image_tok_embeds,
                    decouple_label_tok_embeds=args.decouple_label_tok_embeds,
                    is_train=False,
                    masked_input=masked_input,
                )

                tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
                next_tokens = tokens[:, -1].unsqueeze(1).long()

            # finalize the tokens and logprobs
            tokens, sum_logprobs = text_decoder.finalize(tokens, sum_logprobs)

            # wrap up
            pred_probs = torch.nested.as_nested_tensor(
                [torch.tensor(p) for p in sum_logprobs]
            ).to(device)
            pred_tokens = torch.nested.as_nested_tensor(
                [torch.tensor(t) for t in tokens]
            ).to(device)

            # convert tokens to labels
            batch_preds: List[List[str]] = []
            batch_probs: List[List[float]] = []

            for i in range(bs):
                current_probs = pred_probs[i]
                current_tokens = pred_tokens[i]

                probs_per_label = []
                token_per_label = []

                current_pred_tokens = defaultdict(list)
                current_pred_labels = defaultdict(list)

                # group tokens by the dilimiter
                for prob, token in zip(current_probs, current_tokens):
                    if token != 29892:  # delimiter ","
                        probs_per_label.append(prob)
                        token_per_label.append(token.item())
                    else:
                        # include the delimiter score
                        probs_per_label.append(prob)
                        token_per_label.append(token.item())

                        # compute the final score
                        probs = torch.stack(probs_per_label)
                        label = tokenizer.decode(token_per_label)

                        current_pred_tokens[label].append(token_per_label)
                        current_pred_labels[label].append(probs)

                        probs_per_label = []
                        token_per_label = []

                current_pred_prob = {}
                for label, tokens in current_pred_tokens.items():
                    probs = current_pred_labels[label]
                    # multiple groups of tokens for the same label
                    # we stack them together and compute the sum for each group
                    probs = torch.stack([p.prod() for p in probs], dim=0)
                    prob_per_label = probs.sum()  # sum over all groups
                    current_pred_prob[label] = prob_per_label.item()

                # higher probability is better
                sorted_current_pred_labels = sorted(
                    current_pred_prob.items(), key=lambda x: x[1], reverse=True
                )
                current_preds, current_scores = [], []
                for v in sorted_current_pred_labels:
                    label, score = v
                    current_preds.append(label.replace(",", ""))  # remove the delimiter
                    current_scores.append(round(score, 5))

                batch_preds.append(current_preds)
                batch_probs.append(current_scores)

            t2 = time.perf_counter()

            batch_preds = batch_preds[0]
            batch_probs = batch_probs[0]

            prev.append(batch_preds)
            # print(batch_preds, batch_probs)
            text_decoder.reset()

            for pred,prob in zip(batch_preds,batch_probs):
                if pred not in all_preds.keys():
                    all_preds[pred] = prob
                elif prob > all_preds[pred]:
                    all_preds[pred] = prob
            
      


        # print(f"final predictions: {prev}")  
        sorted_preds = dict(sorted(all_preds.items(), key=lambda item: item[1], reverse=True))
        return list(sorted_preds.keys())[:10]
        # print(f"final predictions: {sorted_preds}")



if __name__ == "__main__":

    cfg = load_config(["--config", "nxtp_ours/src/configs/config_g3m.py"])
    args = cfg.args
    args = set_dtype(args)
    args.resume_ckpt_path = "nxtp_ours/dev_root/ckpt_epoch_03_iter_0021360.pth"
    args.inference_mode = bool(1)
    args.eval_embedding_model = "bert"
    args.max_gen_len = 64

    # set device
    device = torch.device("cuda")

    # load models
    llama_model, tokenizer, model_args = load_llama(args, device)
    clip_model = load_clip(args, device)
    model = LangClassifier(vision_encoder=clip_model, language_decoder=llama_model)
    model = model.to(device)

    checkpoint = "sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=16,  # Coarser grid
        points_per_batch=64,  # Reduce batch size for coarser segmentation
        pred_iou_thresh=0.8,  # High-quality masks
        stability_score_thresh=0.95,  # Focus on stable masks
        stability_score_offset=0.7,  # Keep default
        crop_n_layers=1,  # Minimal cropping for larger masks
        box_nms_thresh=0.5,  # Reduce overlap for distinct objects
        crop_n_points_downscale_factor=2,  # Balance detail and computation
        min_mask_region_area=500.0,  # Focus on larger objects
        use_m2m=True,  # Merge masks for the same object
    )
# load ckpt
    load_checkpoint(args, model, strict=False)
    model.eval()

    # show params
    total_params = sum(param.numel() for param in model.parameters()) / 1e6
    print(f"total params: {total_params:.2f} M, {total_params / 1e3:.2f} B")

    # Configura il dataset
    root_dir = "nxtp_ours/dev_root/data/coco/coco_valid/00000"  # Percorso alla cartella con i dati
    batch_size = 1

    # Crea il dataset e il DataLoader
    dataset = CustomDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    metrics = {
        "sscr": SemanticFScore(
            model_name=args.eval_embedding_model,
            max_seq_len=args.max_seq_len,
        ).to(device),
    }
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    # Esempio di utilizzo del DataLoader
    for _, batch in pbar:
        jsons = batch["json"]    # Batch di JSON
        texts = batch["text"]    # Batch di testo
        image_paths = batch["img_path"]

        for i in range(batch_size):
            text = texts[i]
            image_path = image_paths[i]

            # Esegui la predizione
            predictions = predict(model,mask_generator,tokenizer,image_path,args,device = "cuda",num_labels=10)
            if predictions == []:
                continue
            ground_truth = encode_cap_to_objs(args,[text])
            ground_truth = list(set(ground_truth[0]))
            # print(f"Predictions: {predictions}")
            vals = {}
            for n, metric in metrics.items():
                metric.update([predictions], [ground_truth])
                r, p, f, vals_std = metric.compute()
                vals[f"{n}_R"] = r.item()
                vals[f"{n}_P"] = p.item()
                vals[f"{n}_F"] = f.item()
                for n_std, v_std in vals_std.items():
                    vals[f"{n}_{n_std}"] = v_std.item()

            pbar.set_postfix(
                **vals,
                refresh=False,
            )

            # logging results
            metric = metrics["sscr"]
            scores: List[List[float]] = metric.scores  # [references [predictions]]
            div_scores: List[List[float]] = metric.div_scores  # [predictions!]

            # Save results to JSON
            results = {
                "image_path": image_path,
                "predictions": predictions,
                "ground_truth": ground_truth,
                "metrics": vals,
            }

            output_file = "evalutation_ours_coco_00000_2.json"
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(results)

            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
