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

from lang_sam import LangSAM
import torchvision.transforms as T

from nxtp_ours.src.evals.engine import encode_cap_to_objs
from nxtp_ours.src.evals.metrics import SemanticFScore
from nxtp_ours.src.functions import load_clip, load_llama
from nxtp_ours.src.models.classifier import LangClassifier
from nxtp_ours.src.utils import load_checkpoint, load_config, set_dtype


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
    
def compute_sam_mask(model_lang_sam : LangSAM, label : str, image_path, masked_input,save_mask=False,save_mask_patches=False):


    if label == "" : return masked_input
    image = Image.open(image_path).convert("RGB")
    image = build_preprocess_sam(224)(image)
    output_dir = "./output/"
    filename = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
    output_dir = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    results = model_lang_sam.predict([image], [label])
    patches_masked = None
    for idx, result in enumerate(results):
        # Salvataggio delle maschere come immagini
        for mask_idx, mask in enumerate(result["masks"]):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))  # Convertire in scala di grigi 8-bit
            mask_path = f"{output_dir}/mask_{idx}_{mask_idx}.png"
            if save_mask:
                mask_img.save(mask_path)

            # Carica l'immagine in bianco e nero
            mask_array = np.array(mask_img) // 255  # Converti in binario (0 o 1)

            # Definisci dimensioni delle patch
            patch_size = 14
            h, w = mask_array.shape
            matrix_height = h // patch_size
            matrix_width = w // patch_size

            # Inizializza matrice di patch
            patch_matrix = np.zeros((matrix_height, matrix_width), dtype=int)

            # Calcola i valori delle patch
            for i in range(matrix_height):
                for j in range(matrix_width):
                    patch = mask_array[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                    patch_value = 1 if np.sum(patch) >= (patch_size * patch_size / 2) else 0
                    patch_matrix[i, j] = patch_value

            # Visualizza la matrice come immagine
            # Trasforma la matrice in un tensore torch

            patch_matrix_tensor = torch.tensor(patch_matrix, dtype=torch.int)
            if patches_masked is None:
                patches_masked = patch_matrix_tensor
            else:
                patches_masked += patch_matrix_tensor
                patches_masked = torch.where(patches_masked > 0, 1, 0)


    if patches_masked is None:
        return masked_input
    elif masked_input is None:
        return patches_masked
    else:
        patches_masked = masked_input + patches_masked
        patches_masked = torch.where(patches_masked > 0, 1, 0)

    if save_mask_patches:
        patch_matrix_tensor = patches_masked
        patch_matrix = patch_matrix_tensor.numpy()
        patch_matrix = (patch_matrix * 255).astype(np.uint8)
        mask_img = Image.fromarray(patch_matrix)
        num = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])
        mask_path = f"{output_dir}/mask_{num}.png"
        mask_img.save(mask_path)

    return patches_masked


def predict(model,model_sam,tokenizer,img_path,args,device = "cuda",num_labels=10):

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
        prev = []
        masked_input = None
        while len(prev) < 10:
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

            # print(f"\ninference time: {(t2 - t1):.3f}s")
            # print(f"top-{num_labels} predictions:")
            
            for pred, prob in zip(batch_preds, batch_probs):
                # print(f"| prob: {prob:.5f} - {pred}")
                if pred not in prev:
                    prev.append(pred)
                    text_decoder.reset()
                    if pred == "animal":
                        break
                    masked_input = compute_sam_mask(model_sam,pred, img_path, masked_input)
                    break


        # print(f"final predictions: {prev}")  
        return prev



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

    model_lang_sam = LangSAM()

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
            predictions = predict(model,model_lang_sam,tokenizer,image_path,args,device = "cuda",num_labels=10)
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

            output_file = "evalutation_ours_coco_00000.json"
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(results)

            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
