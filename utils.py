import torch
import numpy as np
from PIL import Image, ImageOps
import math
import sys
from omegaconf import OmegaConf
import cv2
import matplotlib.pyplot as plt
cmap = plt.colormaps['gray']

from diffusers import EulerAncestralDiscreteScheduler

from pipelines.ensemble_pipeline import StableDiffusionInstructPix2PixEnsemblePipeline
from pipelines.ensemble_sd3_pipeline import StableDiffusion3InstructPix2PixEnsemblePipeline
from pipelines.instructdiffusion_pipeline import InstructDiffusionEnsemblePipeline

MODEL_PATH = {
    "instructpix2pix": "timbrooks/instruct-pix2pix",
    "instructdiffusion": "./checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt",
    "magicbrush": "vinesmsuic/magicbrush-jul7",
    "mgie": "./checkpoints/mgie_7b/unet.pt",
    "ultraedit": "BleachNick/SD3_UltraEdit_freeform",
}

def load_model(args):
    if args.model == 'instructpix2pix':
        pipe = StableDiffusionInstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH[args.model], torch_dtype=torch.float16, safety_checker=None).to('cuda')
    elif args.model == "magicbrush":
        pipe = StableDiffusionInstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH[args.model], torch_dtype=torch.float16, safety_checker=None).to('cuda')
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif args.model == 'ultraedit':
        pipe = StableDiffusion3InstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH[args.model], torch_dtype=torch.float16).to('cuda')
    elif args.model == 'instructdiffusion':
        sys.path.append("./InstructDiffusion/stable_diffusion")
        from InstructDiffusion.stable_diffusion.ldm.util import instantiate_from_config
        config = OmegaConf.load("./InstructDiffusion/configs/instruct_diffusion.yaml")
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(MODEL_PATH[args.model], map_location="cuda")["state_dict"], strict=False)
        model = model.to("cuda").eval()
        pipe = InstructDiffusionEnsemblePipeline(model)
    elif args.model == "mgie":
        pipe = StableDiffusionInstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH["instructpix2pix"], torch_dtype=torch.float16, safety_checker=None).to('cuda')
        from mgie_module import MGIE_module
        pipe.unet.load_state_dict(torch.load(MODEL_PATH[args.model], map_location='cpu'))
        mgie_module = MGIE_module(ckpt_dir="./checkpoints")
        pipe.generate_prompt = mgie_module.generate_prompt
    else:
        raise ValueError(f"Model {args.model} not supported.")
    return pipe
    
def image_resize(image, resolution=512):
    width, height = image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    return image

def rle_decode(rle, shape=(512,512)):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for i in range(0, len(rle), 2):
        start = rle[i]
        length = rle[i + 1]
        mask[start:start + length] = 1
    mask = Image.fromarray(mask.reshape(shape) * 255)
    return mask

def heatmap_visualization(heatmap, size=(512,512)):
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.cpu().detach().numpy()
    heatmap = (cmap(heatmap)[:,:,:3] * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, size, cv2.INTER_CUBIC)
    heatmap = Image.fromarray(heatmap)
    return heatmap
