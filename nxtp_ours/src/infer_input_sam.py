from typing import List

import argparse
import os
import time
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from loader import build_preprocess
from models.classifier import LangClassifier
from encoding import construct_text_inputs, construct_embd_inputs
from decoding import OneShotDecoder
from functions import load_llama, load_clip
from utils import load_config, set_dtype, load_checkpoint

from lang_sam import LangSAM
import numpy as np
import torchvision.transforms as T

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def get_last_token_attention_map(attn_map, filename="last_token_attention_map.png", cmap="Blues"):
    last_token_image_attention = attn_map[0, :, -1, :256]  # Shape: [32, 256]

    # Visualizza la mappa di attenzione dell'ultimo token verso i token immagine per ogni testa
    fig, ax = plt.subplots(4, 8, figsize=(16, 8))  # Griglia 4x8 per le 32 teste
    for head_idx in range(last_token_image_attention.shape[0]):
        _map = last_token_image_attention[head_idx].reshape(16, 16)  # Riarrangia in 16x16 per i 256 token
        _map = _map.detach().cpu().numpy()
        _map = (_map - _map.min()) / (_map.max() - _map.min() + 1e-6)  # Normalizzazione
        ax[head_idx // 8, head_idx % 8].imshow(_map, cmap=cmap)
        ax[head_idx // 8, head_idx % 8].axis("off")

    plt.tight_layout()
    os.makedirs("attentions", exist_ok=True)
    plt.savefig(f"attentions/{filename}")
    plt.close()

    # Calcolo della media delle attention map su tutte le teste
    avg_attention = last_token_image_attention.mean(axis=0)  # Shape: [256]
    avg_attention_map = avg_attention.reshape(16, 16)  # Riarrangia in 16x16
    avg_attention_map = avg_attention_map.detach().cpu().numpy()
    avg_attention_map = (avg_attention_map - avg_attention_map.min()) / (
        avg_attention_map.max() - avg_attention_map.min() + 1e-6
    )  # Normalizzazione

    # Visualizza la mappa di attenzione media
    plt.figure(figsize=(8, 8))
    plt.imshow(avg_attention_map, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"attentions/average_{filename}")
    plt.close()


def visualize_attention_map(attn_map, filename="average_attention_map.png", cmap="Blues"):
    """
    Visualizza e salva una mappa di attenzione per un'immagine divisa in patch 16x16.

    Parametri:
        attn_map (torch.Tensor): Tensore di attenzione di dimensioni (256, 256).
        filename (str): Nome del file in cui salvare l'immagine.
        cmap (str): Mappa di colori da utilizzare per la visualizzazione.
    """
    # Verifica che attn_map sia di dimensioni corrette
    if attn_map.shape != (256, 256):
        raise ValueError("Il tensore di attenzione deve avere dimensioni (256, 256).")

    # Calcola la media delle attenzioni lungo la dimensione delle righe o colonne
    # (opzionale, a seconda dell'analisi desiderata)
    avg_attn_map = attn_map.mean(dim=0)  # Otteniamo un vettore di dimensioni (256,)

    # Riorganizza la mappa di attenzione media in una griglia 16x16
    avg_attn_grid = avg_attn_map.reshape(16, 16).detach().cpu().numpy()

    # Normalizza la mappa per una migliore visualizzazione
    avg_attn_grid = (avg_attn_grid - avg_attn_grid.min()) / (avg_attn_grid.max() - avg_attn_grid.min() + 1e-6)

    # Visualizza la mappa di attenzione
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_attn_grid, cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    
    # Salva la mappa di attenzione
    plt.savefig(filename)
    plt.close()

def save_attention_mean_layer(heads_attention, filename="average_attention_map.png"):
    # attention head index: 0-31
    ii = 0
    ij = 256
    heads_attention = heads_attention[:, :, ii:ij, ii:ij]
    mean_attention = heads_attention.mean(dim=1)
    fig, ax = plt.subplots(16, 16, figsize=(11, 11))
    mean_attention = mean_attention[0]
    # mean_attention = mean_attention/mean_attention.sum()
    # visualize_attention_map(mean_attention)
    total_sum = torch.zeros((16, 16))
    for i in range(256):
        _map = mean_attention[i].reshape(16, 16)
        _map = _map.detach().cpu().numpy()
        _map = (_map - _map.min()) / (_map.max() - _map.min() + 1e-6)
        total_sum += _map
        ax[i // 16, i % 16].imshow(_map, cmap="Blues")
        ax[i // 16, i % 16].axis("off")
    total_sum /= 256

    plt.figure(figsize=(6, 6))
    plt.imshow(total_sum, cmap="Blues")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"total/{filename}")
    plt.close()
    
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"total2/{filename}")
    plt.close()


    return total_sum

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


@torch.inference_mode()
def main(
    ckpt_path: str,
    img_path: str,
    num_labels: int = 10,
    save_attention_map: bool = False,
):
    # load config
    cfg = load_config(["--config", "configs/config_g3m.py"]).args
    cfg = set_dtype(cfg)
    cfg.resume_ckpt_path = ckpt_path
    cfg.inference_mode = bool(1)

    # set device
    device = torch.device("cuda")

    # load models
    llama_model, tokenizer, model_args = load_llama(cfg, device)
    clip_model = load_clip(cfg, device)
    model = LangClassifier(vision_encoder=clip_model, language_decoder=llama_model)
    model = model.to(device)

    model_lang_sam = LangSAM()

    # load ckpt
    load_checkpoint(cfg, model, strict=False)
    model.eval()

    # show params
    total_params = sum(param.numel() for param in model.parameters()) / 1e6
    print(f"total params: {total_params:.2f} M, {total_params / 1e3:.2f} B")

    # ctx manager
    ctx = torch.amp.autocast(device_type="cuda", dtype=cfg.ptdtype)

    # load image
    img = Image.open(img_path).convert("RGB")
    img = build_preprocess(cfg.input_size)(img)
    img = img.unsqueeze(0).to(device)
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
            cfg, caps, tokenizer, offset=n_img_tokens, is_train=False
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
        cached_tensors = dict() if save_attention_map else None

        # start sampling
        x = input_embds
        prev = []
        # Load tensor from .pt file
        # loaded_tensor = torch.load(tensor_path)
        # print(f"Loaded tensor from {tensor_path}: {loaded_tensor}")
        # masked_input = masked_input.T
        all_preds = {}

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
                prefix_image_tok_embeds=cfg.prefix_image_tok_embeds,
                decouple_label_tok_embeds=cfg.decouple_label_tok_embeds,
                is_train=False,
                cached_tensors=cached_tensors,
                masked_input=masked_input,
            )
            total = torch.zeros((16, 16))
            if save_attention_map:
                for k in cached_tensors.keys():
                    if not "attn" in k:
                        continue
                    # visualize relatively shallow layers in the decoder
                    # if not "layer_idx_0" in k:
                    #     continue

                    # print(f"visualizing attention map for {k}")
                    attn_map = cached_tensors[k]
                    # get_last_token_attention_map(attn_map, filename=f"pred{len(prev)}_layer{k}.png")
                    continue
                    # extract the attention map for image tokens
                    ii = dummy_token_index_obj
                    ij = dummy_token_index_obj + n_img_tokens
                    attn_map = attn_map[:, :, ii:ij, ii:ij]

                    # attention head index: 0-31
                    for head_idx in tqdm(range(attn_map.shape[1]), leave=False):
                        # save attention map
                        fig, ax = plt.subplots(16, 16, figsize=(11, 11))
                        maps = attn_map[0, head_idx]
                        for i in range(attn_map.shape[2]):
                            _map = maps[i].reshape(16, 16)
                            _map = _map.detach().cpu().numpy()
                            _map = (_map - _map.min()) / (_map.max() - _map.min() + 1e-6)
                            ax[i // 16, i % 16].imshow(_map, cmap="Blues")
                            ax[i // 16, i % 16].axis("off")
                        plt.tight_layout()
                        os.makedirs("figs", exist_ok=True)
                        plt.savefig(f"figs/attn_map_{k}_head_idx_{head_idx}.png")
                        plt.close()

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
                    prefix_image_tok_embeds=cfg.prefix_image_tok_embeds,
                    decouple_label_tok_embeds=cfg.decouple_label_tok_embeds,
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

            print(f"\ninference time: {(t2 - t1):.3f}s")
            print(f"top-{num_labels} predictions:")

            prev.append(batch_preds)
            print(batch_preds, batch_probs)
            text_decoder.reset()

            for pred,prob in zip(batch_preds,batch_probs):
                if pred not in all_preds.keys():
                    all_preds[pred] = prob
                elif prob > all_preds[pred]:
                    all_preds[pred] = prob

        sorted_preds = dict(sorted(all_preds.items(), key=lambda item: item[1], reverse=True))
        print(f"final predictions: {sorted_preds}")
        print(list(sorted_preds.keys())[:10])
    pass

def build_preprocess_sam(input_size):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    # NOTE: because we freeze CLIP, won't apply augmentations on images for now
    resized_crop = [
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
    ]

    return T.Compose([*resized_crop, *to_rgb])

def compute_sam_mask(model_lang_sam : LangSAM, label : str, image_path, masked_input):

    image = Image.open(image_path).convert("RGB")
    image = build_preprocess_sam(224)(image)
    output_dir = "./output/"
    filename = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
    output_dir = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    save_mask = True
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

    if save_mask:
        patch_matrix_tensor = patches_masked
        patch_matrix = patch_matrix_tensor.numpy()
        patch_matrix = (patch_matrix * 255).astype(np.uint8)
        mask_img = Image.fromarray(patch_matrix)
        num = len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])
        mask_path = f"{output_dir}/mask_{num}.png"
        mask_img.save(mask_path)

    return patches_masked

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--img-path", type=str, required=True)
    parser.add_argument("--num-labels", type=int, default=10)
    parser.add_argument("--save-attention-map", type=bool, default=False)
    args = parser.parse_args()

    main(args.ckpt_path, args.img_path, args.num_labels, args.save_attention_map)
