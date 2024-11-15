from PIL import Image
from lang_sam import LangSAM
import os
import pandas as pd
import json
import numpy as np

model = LangSAM()
image_pil = Image.open("nxtp/assets/totk.jpg").convert("RGB")
text_prompt = "sky."
results = model.predict([image_pil], [text_prompt])

# Directory for output files
output_dir = "./output/"
os.makedirs(output_dir, exist_ok=True)

# Process and save each result
for idx, result in enumerate(results):
    # Save boxes and scores as CSV
    boxes_df = pd.DataFrame(result["boxes"], columns=["x1", "y1", "x2", "y2"])
    scores_df = pd.DataFrame(result["scores"], columns=["score"])
    boxes_df.to_csv(f"{output_dir}/boxes_{idx}.csv", index=False)
    scores_df.to_csv(f"{output_dir}/scores_{idx}.csv", index=False)

    # Save masks as images
    for mask_idx, mask in enumerate(result["masks"]):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to 8-bit grayscale
        mask_img.save(f"{output_dir}/mask_{idx}_{mask_idx}.png")
        

    # Save mask scores as JSON
    mask_scores = result["mask_scores"].tolist()
    with open(f"{output_dir}/mask_scores_{idx}.json", "w") as f:
        json.dump(mask_scores, f)