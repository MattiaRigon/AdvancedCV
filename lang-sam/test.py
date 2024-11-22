from PIL import Image
from lang_sam import LangSAM
import os
import pandas as pd
import json
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

def build_preprocess(input_size):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    # NOTE: because we freeze CLIP, won't apply augmentations on images for now
    resized_crop = [
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
    ]

    return T.Compose([*resized_crop, *to_rgb])

if __name__ == "__main__":

    model = LangSAM()
    filename = "animals"
    image_pil = Image.open(f"nxtp/assets/{filename}.png").convert("RGB")
    image_pil = build_preprocess(224)(image_pil)

    text_prompt = "photo."
    results = model.predict([image_pil], [text_prompt])

    # Directory for output files
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)

    # Process and save each result


    for idx, result in enumerate(results):
        # Salvataggio di box e punteggi come CSV
        boxes_df = pd.DataFrame(result["boxes"], columns=["x1", "y1", "x2", "y2"])
        scores_df = pd.DataFrame(result["scores"], columns=["score"])
        boxes_df.to_csv(f"{output_dir}/boxes_{idx}.csv", index=False)
        scores_df.to_csv(f"{output_dir}/scores_{idx}.csv", index=False)

        # Salvataggio delle maschere come immagini
        for mask_idx, mask in enumerate(result["masks"]):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))  # Convertire in scala di grigi 8-bit
            mask_path = f"{output_dir}/mask_{idx}_{mask_idx}.png"
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

            # Salva il tensore su un file
            torch.save(patch_matrix_tensor, f"{output_dir}/patch_matrix_{idx}_{mask_idx}_{filename}.pt")
            plt.figure(figsize=(6, 6))
            plt.imshow(patch_matrix, cmap="gray", interpolation="nearest")
            plt.title(f"Patch Matrix for Mask {mask_idx}")
            plt.axis("off")
            plt.savefig(f"{output_dir}/patch_matrix_{idx}_{mask_idx}.png")

        # Salvataggio dei punteggi delle maschere come JSON
        mask_scores = result["mask_scores"].tolist()
        with open(f"{output_dir}/mask_scores_{idx}.json", "w") as f:
            json.dump(mask_scores, f)
