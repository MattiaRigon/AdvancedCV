from collections import defaultdict
import os
import json
from typing import List
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn.functional as F
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

@torch.inference_mode()
def generate(
    args,
    n_img_tokens,
    dummy_token_index,
    input_tokens,
    input_embds,
    input_tokens_objs,
    tokenizer,
    model,
    device,
    text_decoder,
):
    # because all input samples have the same prompt and the same length of image embeddings,
    # here we only need to use one index for shaving out padding ids for all
    shave_ind = torch.where(input_tokens == tokenizer.pad_id)[1][0]

    min_prompt_size = shave_ind.item()
    max_prompt_size = shave_ind.item()

    total_len = min(args.max_seq_len, args.max_gen_len + max_prompt_size)

    bs = input_tokens.shape[0]
    Wte = model.language_decoder.tok_embeddings.weight

    input_embds = input_embds[:, : shave_ind + n_img_tokens]
    tokens = input_tokens[:, :shave_ind]  # will be final output tokens
    sum_logprobs = torch.zeros(bs, device=device)
    input_tokens_objs = input_tokens_objs[:, : shave_ind + n_img_tokens]

    if args.text_decoder_strategy == "beam":
        # repeat the input for beam search: [bs, n_embd] -> [bs * n_beam, n_embd]
        n_beam = args.beam_size
        input_embds = input_embds.repeat_interleave(n_beam, dim=0)
        tokens = tokens.repeat_interleave(n_beam, dim=0)
        sum_logprobs = sum_logprobs.repeat_interleave(n_beam, dim=0)
        input_tokens_objs = input_tokens_objs.repeat_interleave(n_beam, dim=0)

    pred_probs = []
    pred_tokens = []

    start_pos = min_prompt_size
    for cur_pos in range(start_pos, total_len):
        if cur_pos == start_pos:
            x = input_embds
        else:
            x = torch.cat(
                [
                    x,
                    Wte[next_tokens.long()],
                ],
                dim=1,
            )

        logits = model.language_decoder.forward(
            x,
            start_pos=0,
            dummy_token_index=dummy_token_index,
            offset=n_img_tokens,
            input_tokens=input_tokens_objs,
            prefix_image_tok_embeds=args.prefix_image_tok_embeds,
            decouple_label_tok_embeds=args.decouple_label_tok_embeds,
            is_train=False,
        )

        tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
        next_tokens = tokens[:, -1].unsqueeze(1)
        prob_next_tokens = text_decoder.prob_next_tokens

        # NOTE: shape of
        # tokens: torch.Size([64, 10]) -> torch.Size([64, 11]), ...
        # prob_next_tokens: torch.Size([64, 1])
        # next_tokens: torch.Size([64, 1])

        if args.text_decoder_strategy == "one_shot" and cur_pos == start_pos:
            # here only samples the first set of tokens
            input_tokens_objs = torch.cat(
                [
                    input_tokens_objs.repeat_interleave(
                        text_decoder.one_shot_size, dim=0
                    ),
                    next_tokens,
                ],
                dim=1,
            )
            break

        if args.text_decoder_strategy == "greedy":
            # update input tokens for updating attention mask
            input_tokens_objs = torch.cat([input_tokens_objs, next_tokens], dim=1)
            pred_tokens.append(next_tokens)
            pred_probs.append(prob_next_tokens)

        if completed:
            break

    if args.text_decoder_strategy == "one_shot":
        while completed == False:
            if x.shape[0] != next_tokens.shape[0]:
                assert next_tokens.shape[0] % x.shape[0] == 0
                x = x.repeat_interleave(next_tokens.shape[0] // x.shape[0], dim=0)

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
                dummy_token_index=dummy_token_index,
                offset=n_img_tokens,
                input_tokens=input_tokens_objs,
                prefix_image_tok_embeds=args.prefix_image_tok_embeds,
                decouple_label_tok_embeds=args.decouple_label_tok_embeds,
                is_train=False,
            )

            tokens, completed = text_decoder.update(tokens, logits, sum_logprobs)
            next_tokens = tokens[:, -1].unsqueeze(1).long()

    tokens, sum_logprobs = text_decoder.finalize(tokens, sum_logprobs)

    if args.text_decoder_strategy == "one_shot":
        pred_probs = torch.nested.as_nested_tensor(
            [torch.tensor(p) for p in sum_logprobs]
        ).to(device)
        pred_tokens = torch.nested.as_nested_tensor(
            [torch.tensor(t) for t in tokens]
        ).to(device)
    elif args.text_decoder_strategy == "beam":
        pred_tokens = torch.stack([torch.tensor(t) for t in tokens], dim=0).to(device)
        pred_probs = torch.zeros_like(pred_tokens).to(device)
    elif args.text_decoder_strategy == "greedy":
        pred_probs = torch.cat(pred_probs, dim=1)
        pred_tokens = torch.cat(pred_tokens, dim=1)
    else:
        raise NotImplementedError

    return pred_probs, pred_tokens


def post_process(
    pred_probs,
    pred_tokens,
    embds_imgs,
    tokenizer,
    Wte,
    ranking_method="sim",
    sim_reduction="mean",
    sim_euclidean_dist=False,
    sim_max_reduction_with_fscore=False,
):
    assert ranking_method in ("sim", "ppl", "prob", "clip", "none")
    assert sim_reduction in ("mean", "max")

    if ranking_method == "sim":
        # check the settings
        if sim_reduction == "mean" and sim_euclidean_dist is False:
            raise ValueError("cannot use mean reduction for cosine similarity")

    # NOTE: the following assert is not available for nested tensors
    #       so just ignore it for now
    # assert pred_probs.shape == pred_tokens.shape

    bs = embds_imgs.shape[0]

    batch_preds: List[List[str]] = []
    batch_probs: List[List[float]] = []

    for i in range(bs):
        current_embds_imgs = embds_imgs[i]
        current_probs = pred_probs[i]
        current_tokens = pred_tokens[i]

        probs_per_label = []
        token_per_label = []

        current_pred_tokens = defaultdict(list)
        current_pred_labels = defaultdict(list)

        # step 1: group tokens by the dilimiter
        for prob, token in zip(current_probs, current_tokens):
            if token != 29892:
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

        # step 2: compute the similarity between image tokens and label tokens
        if ranking_method == "sim":
            # Eq. A.4 and A.5 in the paper
            current_pred_sim = {}
            for label, tokens in current_pred_tokens.items():
                sim_per_label = []
                for group_tokens in tokens:
                    embds_label = torch.stack([Wte[t] for t in group_tokens], dim=0)
                    v = F.normalize(current_embds_imgs, dim=-1)  # [m, d]
                    t = F.normalize(embds_label, dim=-1)  # [n, d]
                    M = torch.einsum("nd,md->nm", t, v)  # [n, m]

                    if sim_euclidean_dist:
                        # euclidean distance in [0, 1], [sim, dissim]
                        M = torch.sqrt(2 - 2 * M) / 2
                        sim_reverse = False
                    else:
                        # cosine similarity in [-1, 1], [dissim, sim]
                        sim_reverse = True

                    if sim_reduction == "max":
                        Rt = M.max(dim=1).values.mean()
                        if sim_max_reduction_with_fscore:
                            Pi = M.max(dim=0).values.mean()
                            sim_score = 2 * Pi * Rt / (Pi + Rt)
                        else:
                            sim_score = Rt
                    elif sim_reduction == "mean":
                        sim_score = M.mean()
                    else:
                        raise NotImplementedError
                    sim_per_label.append(sim_score)

                # multiple groups of tokens for the same label
                # we stack them together and compute the mean for each label
                sim_per_label = torch.stack(sim_per_label).mean()
                current_pred_sim[label] = sim_per_label.item()

            # higher value means more similar for cosine similarity
            # lower value means more similar for euclidean distance
            sorted_current_pred_labels = sorted(
                current_pred_sim.items(), key=lambda x: x[1], reverse=sim_reverse
            )
        elif ranking_method == "ppl":
            # Eq. A.3 in the paper
            current_pred_ppl = {}
            for label, tokens in current_pred_tokens.items():
                probs = current_pred_labels[label]
                # multiple groups of tokens for the same label
                # we stack them together and select the one with the lowest ppl
                ppls = torch.stack([p.log().mean(dim=-1).exp() for p in probs], dim=0)
                ppl_per_label = ppls.min()  # min over all groups
                current_pred_ppl[label] = ppl_per_label.item()

            # lower perplexity is better
            sorted_current_pred_labels = sorted(
                current_pred_ppl.items(), key=lambda x: x[1], reverse=False
            )
        elif ranking_method == "prob":
            # Eq. A.2 in the paper
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
        elif ranking_method == "clip":
            # Eq. A.1 in the paper
            current_pred_clip_score = {}
            for label, tokens in current_pred_tokens.items():
                current_pred_clip_score[label] = (
                    0.0  # will have it later in the evals.engine function
                )
            sorted_current_pred_labels = sorted(
                current_pred_clip_score.items(), key=lambda x: x[1], reverse=True
            )
        elif ranking_method == "none":
            current_pred_none_score = {}
            for label, tokens in current_pred_tokens.items():
                current_pred_none_score[label] = 0.0
            # keep the original order without sorting
            sorted_current_pred_labels = current_pred_none_score.items()
        else:
            raise NotImplementedError

        current_preds, current_scores = [], []
        for v in sorted_current_pred_labels:
            label, score = v
            current_preds.append(label.replace(",", ""))  # remove the delimiter
            current_scores.append(round(score, 5))

        batch_preds.append(current_preds)
        batch_probs.append(current_scores)

    return batch_preds, batch_probs


@torch.inference_mode()
def engine_lang_classifier(args, preprocess, model, tokenizer, imgs, paths=None):

    device = "cuda"
    img = Image.open(imgs[0]).convert("RGB")
    img = build_preprocess(args.input_size)(img)
    img = img.unsqueeze(0).to(device)


    k = 10
    text_decoder = OneShotDecoder(k=k)

    # call reset() to initialize the decoder
    text_decoder.reset()

    t1 = time.perf_counter()

    h = model.encode_images(img)
    z = model.decode_images(h)

    embds_clss, embds_imgs = z[:, :1], z[:, 1:]
    bs, n_img_tokens = embds_imgs.shape[:2]

    caps = ["" for _ in range(bs)]
    (
        tokens_caps,
        tokens_objs,
        _,
        _,
        dummy_token_index_cap,
        dummy_token_index_obj,
    ) = construct_text_inputs(
        args,
        caps,
        tokenizer,
        offset=n_img_tokens,
        is_train=False,
        return_strs=False,
        skip_extract_nouns=args.skip_extract_nouns,
    )
    tokens_caps = tokens_caps.to(device)
    tokens_objs = tokens_objs.to(device)

    Wte = model.language_decoder.tok_embeddings.weight
    embds_caps = Wte[tokens_caps]
    embds_objs = Wte[tokens_objs]

    input_embds_caps, input_embds_objs, input_tokens_objs = construct_embd_inputs(
        embds_imgs,
        embds_caps,
        embds_objs,
        dummy_token_index_cap,
        dummy_token_index_obj,
        tokens_caps,
        tokens_objs,
        tokenizer,
    )

    pred_probs, pred_tokens = generate(
        args,
        n_img_tokens,
        dummy_token_index_obj,
        tokens_objs,
        input_embds_objs,
        input_tokens_objs,
        tokenizer,
        model,
        device,
        text_decoder,
    )

    objs = post_process(
        pred_probs,
        pred_tokens,
        embds_imgs,
        tokenizer,
        Wte,
        ranking_method='prob',
        sim_reduction="mean",
        sim_euclidean_dist=True,
        sim_max_reduction_with_fscore=False,
    )


    return objs

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
            predictions,probabilities = engine_lang_classifier(args, None, model, tokenizer, [image_path])
            predictions = predictions[0]
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

            output_file = "evalutation_nxtp_coco_00000.json"
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(results)

            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
