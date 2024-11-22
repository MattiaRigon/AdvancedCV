import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as T


def visualize_attention_transparency(img, attention_matrix, save_path=None):
    """
    Visualize attention values as transparency masks over an image.
    
    Args:
        image_path (str): Path to the original image
        attention_matrix (numpy.ndarray): 16x16 attention matrix with values between 0 and 1
        save_path (str, optional): Path to save the visualization. If None, displays the plot
    """
    img = np.array(img)
    # Get image dimensions
    h, w = img.shape[:2]
    patch_h, patch_w = h // 16, w // 16
    
    # Create output image with alpha channel
    output = np.zeros((h, w, 4), dtype=np.uint8)
    output[..., :3] = img
    
    # Create alpha mask
    alpha_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Normalize attention matrix to 0-255 range for alpha values
    attention_matrix = (attention_matrix - attention_matrix.min()) / (attention_matrix.max() - attention_matrix.min())
    attention_matrix = (attention_matrix * 255).astype(np.uint8)
    
    # Apply attention values to patches
    for i in range(16):
        for j in range(16):
            y_start = i * patch_h
            y_end = (i + 1) * patch_h
            x_start = j * patch_w
            x_end = (j + 1) * patch_w
            
            alpha_mask[y_start:y_end, x_start:x_end] = attention_matrix[i, j]
    
    # Set alpha channel
    output[..., 3] = alpha_mask
    
    # Create figure and display
    plt.figure(figsize=(10, 10))
    plt.imshow(output)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    else:
        plt.show()
    plt.close()

def create_sample_visualization():
    """
    Create a sample visualization using random attention values
    """
    # Create sample attention matrix (16x16)
    attention_matrix = np.random.rand(16, 16)
    
    image_pil = Image.open(f"nxtp_ours/dev_root/data/coco/coco_valid/00000/000000009400.jpg").convert("RGB")
    image_pil = build_preprocess(224)(image_pil)
    
    # Visualize
    visualize_attention_transparency(image_pil, attention_matrix, 'attention_visualization.png')

def build_preprocess(input_size):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    # NOTE: because we freeze CLIP, won't apply augmentations on images for now
    resized_crop = [
        # https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
    ]

    return T.Compose([*resized_crop, *to_rgb])

# Example usage
if __name__ == "__main__":
    # For demonstration with sample data
    create_sample_visualization()
    
    # For actual use with your data:
    # attention_matrix = your_16x16_attention_matrix
    # visualize_attention_transparency('path_to_your_image.jpg', attention_matrix)