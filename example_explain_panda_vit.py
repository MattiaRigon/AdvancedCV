from atman_magma.explainer import Explainer
from atman_magma.logit_parsing import get_delta_cross_entropies
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from nxtp.src.functions import load_clip, load_llama
from nxtp.src.utils import load_config, set_dtype


print('loading model...')
cfg = load_config(["--config", "configs/config_g3m.py"]).args
cfg = set_dtype(cfg)
cfg.resume_ckpt_path = "nxtp/nxtp/ckpt_epoch_03_iter_1656549.pth"
cfg.inference_mode = bool(1)

# set device
device = torch.device("cuda")

# load models
llama_model, tokenizer, model_args = load_llama(cfg, device)
clip_model = load_clip(cfg, device)

# from magma.image_input import ImageInput
import  PIL.Image as PilImage


ex = Explainer(
    model = clip_model,
    device = device,
    tokenizer = model.tokenizer,
#     conceptual_suppression_threshold = None
    conceptual_suppression_threshold = 0.75
)

prompt =[
    ## supports urls and path/to/image
    PilImage.open('openimages-panda.jpg'),
    'This is a picture of a'
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(prompt.copy())

label ='Panda'
logit_outputs = ex.collect_logits_by_manipulating_attention(
    prompt = prompt.copy(),
    target = label,
    max_batch_size=1,
    # prompt_explain_indices=[i for i in range(10)]
)

results = get_delta_cross_entropies(
    output = logit_outputs
)

image_filename = 'openimages-panda.jpg'

label_tokens =  model.tokenizer.encode(label)

image = np.zeros((12,12))
for i in range(len(label_tokens)):
    image += results.show_image(image_token_start_idx = 0, target_token_idx= i) **2

# image[image<0.6]=1.0
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (15 , 6))
title = f''
fig.suptitle(title)
ax[0].imshow(cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB))
ax[1].imshow(image)

fig.savefig('panda-explained-atman.jpg')
print('panda-explained-atman.jpg')
