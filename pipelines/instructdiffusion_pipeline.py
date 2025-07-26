import torch
import torch.nn as nn
from torch import autocast
import einops
from einops import rearrange
import k_diffusion as K
import math
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

class InstructDiffusionEnsemblePipeline(nn.Module):
    """
    InstructDiffusionEnsemblePipeline
    This class is a custom pipeline for ensemble inference using InstructDiffusion.
    Reimplemented based on https://github.com/cientgu/InstructDiffusion
    Main Features:
        - Supports multiple candidate seeds for ensemble generation.
        - Selects the best seed based on background inconsistency score (BIS).
        - Accumulates a relevance map for mask extraction during inference.
        - Provides visualization options for all seeds and the best seed.
    Methods:
        get_best_output_from_various_seeds:
            Runs the diffusion process with dynamic stopping and ensemble seed selection.
            Returns a dictionary containing output images for each seed, the best seed, and the relevance mask.
        accumulate_relevance_map:
            Accumulates and normalizes the relevance map based on noise differences at each step.
        calculate_BIS:
            Calculates the background inconsistency score between the input image and the generated image.
        get_noise_diff:
            Computes the normalized noise difference between conditional and unconditional noise predictions,
            with outlier removal.
    """
    def __init__(self, model):
        super().__init__()
        model_wrap = K.external.CompVisDenoiser(model)
        self.null_token = model.get_learned_conditioning([""])
        self.inner_model = model_wrap
        self.model = model

    def forward(
        self, 
        args, 
        prompt = None,
        input_image = None, 
        s_noise = 1.0, 
        eta = 1.0
    ):
        width, height = input_image.size
        factor = args.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width_resize = int((width * factor) // 64) * 64
        height_resize = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)
        
        with torch.no_grad(), autocast("cuda"):
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning([prompt])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(next(self.model.parameters()).device)
            cond["c_concat"] = [self.model.encode_first_stage(input_image).mode()]
            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.text_cfg_scale,
            "image_cfg_scale": args.image_cfg_scale,
        }

        with torch.no_grad(), autocast("cuda"):
            sigmas = self.inner_model.get_sigmas(args.inference_steps)

            torch.manual_seed(args.seed)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            
            """Ancestral sampling with Euler method steps."""
            noise_sampler = K.sampling.default_noise_sampler(z)
            s_in = z.new_ones([z.shape[0]])
            
            for i in trange(len(sigmas) - 1, disable=None):
                sigma = sigmas[i] * s_in
                cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
                cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
                cfg_cond = {
                    "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
                    "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
                }
                out_cond, out_img_cond, out_txt_cond \
                    = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
                denoised = 0.5 * (out_img_cond + out_txt_cond) + \
                        args.text_cfg_scale * (out_cond - out_img_cond) + \
                        args.image_cfg_scale * (out_cond - out_txt_cond)
                sigma_down, sigma_up = K.sampling.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
                d = K.sampling.to_d(z, sigmas[i], denoised)
                # Euler method
                dt = sigma_down - sigmas[i]
                z = z + d * dt
                if sigmas[i + 1] > 0:
                    z = z + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

            x = self.model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")

            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image = ImageOps.fit(edited_image, (width, height), method=Image.Resampling.LANCZOS)
        
        res = {
            "output": edited_image
        }
        return res

    def get_best_output_from_various_seeds(
        self, 
        image = None,
        prompt = None,
        num_inference_steps = 100,
        stopping_step=40,
        first_step_for_mask_extraction=0,
        last_step_for_mask_extraction=20,
        s_noise = 1.0, 
        eta = 1.0,
        args = None,
    ):
        """
        This function is reconstructed based on the InstructDiffusion repository (https://github.com/cientgu/InstructDiffusion).`
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image`):
                `Image` representing an image batch to be repainted according to `prompt`.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
                
            stopping_step (`int`, *optional*, defaults to 40): 
                *Argument for ELECT*. The step at which to select the best seed.
            first_step_for_mask_extraction (`int`, *optional*, defaults to 0):
                *Argument for ELECT*. The first step for relevance mask extraction. This is used to accumulate the relevance map.
            last_step_for_mask_extraction (`int`, *optional*, defaults to 20):
                *Argument for ELECT*. The last step for relevance mask extraction. This is used to accumulate the relevance map.
                
            s_noise (`float`, *optional*, defaults to 1.0):
                The noise scale for the diffusion process.
            eta (`float`, *optional*, defaults to 1.0):
                The eta parameter for the diffusion process, controlling the noise addition during sampling.
                
            args:
                Arguments from the command line, including:
                - `resolution`: The resolution of the output image.
                - `candidate_seeds`: A list of candidate seeds for ensemble generation.
                - `text_cfg_scale`: The text guidance scale for the diffusion model.
                - `image_cfg_scale`: The image guidance scale for the diffusion model.
            
        Returns:
            dict: A dictionary containing:
                - "output-{seed}": PIL.Image for each seed's output. (if args.visualize_all_seeds is False, only the best seed is returned)
                - "output-best_seed": PIL.Image for the best seed (if args.visualize_all_seeds is True).
                - "output_mask": PIL.Image of the mean relevance map.
        """
        
        width, height = image.size
        factor = args.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width_resize = int((width * factor) // 64) * 64
        height_resize = int((height * factor) // 64) * 64
        image = ImageOps.fit(image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)
        
        with torch.no_grad(), autocast("cuda"):
            cond = {}
            cond["c_crossattn"] = [self.model.get_learned_conditioning(prompt)]
            image = 2 * torch.tensor(np.array(image)).float() / 255 - 1
            image = rearrange(image, "h w c -> 1 c h w").to(next(self.model.parameters()).device)
            cond["c_concat"] = [self.model.encode_first_stage(image).mode()]
            uncond = {}
            uncond["c_crossattn"] = [self.null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.text_cfg_scale,
            "image_cfg_scale": args.image_cfg_scale,
        }

        decoded_input = self.model.decode_first_stage(cond["c_concat"][0])
        input_image_np = decoded_input.detach().cpu().numpy()

        candidate_seeds = args.candidate_seeds
        active_seeds = list(candidate_seeds)
        num_seeds = len(candidate_seeds)
        
        seed_latents_dict = {}
        for idx, seed in enumerate(candidate_seeds):
            sigmas = self.inner_model.get_sigmas(num_inference_steps)
            torch.manual_seed(seed)
            initial_latents = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            noise_sampler = K.sampling.default_noise_sampler(initial_latents)
            s_in = initial_latents.new_ones([initial_latents.shape[0]]) 
            seed_latents_dict[seed] = {
                "latents": initial_latents,
                "noise_sampler": noise_sampler,
                "s_in": s_in,
                "idx": idx
            }
        
        self.mean_relevance_map = None
        self.first_step_for_mask_extraction = first_step_for_mask_extraction
        self.last_step_for_mask_extraction = last_step_for_mask_extraction

        current_z0ts = {} # z0t (predicted x0) for each seed at stopping step
        last_outputs = {}
        last_z0s = {}
        
        all_noise_diff_loss = []
        with torch.no_grad(), autocast("cuda"):
            for i in tqdm(range(len(sigmas) - 1)):
                eps_diffs = [] # noise differences for each seed
                for seed in active_seeds:
                    tmp_dict = seed_latents_dict[seed]
                    z = tmp_dict["latents"]
                    noise_sampler = tmp_dict["noise_sampler"]
                    s_in = tmp_dict["s_in"]
                    
                    sigma = sigmas[i] * s_in
                    cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
                    cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
                    cfg_cond = {
                        "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
                        "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
                    }
                    out_cond, out_img_cond, out_txt_cond \
                        = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
                    denoised = 0.5 * (out_img_cond + out_txt_cond) + \
                            args.text_cfg_scale * (out_cond - out_img_cond) + \
                            args.image_cfg_scale * (out_cond - out_txt_cond)
                    z0t = denoised
                    sigma_down, sigma_up = K.sampling.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
                    d = K.sampling.to_d(z, sigmas[i], denoised)
                    # Euler method
                    dt = sigma_down - sigmas[i]
                    z = z + d * dt
                    if sigmas[i + 1] > 0:
                        z = z + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

                    eps_diffs.append(self.get_noise_diff(out_cond, out_img_cond))
                    
                    seed_latents_dict[seed]["latents"] = z # update latents of each seed
                    
                    if i == stopping_step - 1:
                        current_z0ts[seed] = z0t
                        
                self.accumulate_relevance_map(eps_diffs, current_step = i)
                
                
                # selecting best seed at stopping step
                if i == stopping_step - 1:
                    background_inconcistency_scores = {}
                    for seed in candidate_seeds:
                        z0_image = self.model.decode_first_stage(current_z0ts[seed])
                        z0_image_np = z0_image.detach().cpu().numpy()
                        background_inconcistency_scores[seed] = self.calculate_BIS(input_image_np, z0_image_np)
                    best_seed, best_score = min(background_inconcistency_scores.items(), key=lambda x: x[1])
                            
                    # After selecting the best seed, remove the rest
                    # If visualize_all_seeds is True, keep all seeds for visualization
                    if not args.visualize_all_seeds:
                        active_seeds = [best_seed]
                       
            for seed in active_seeds:
                latents = seed_latents_dict[seed]["latents"]
                z0_image = self.model.decode_first_stage(latents)
                last_z0s[seed] = z0_image
        
        return_dict = {}

        for seed, img in last_z0s.items():
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            img = 255.0 * rearrange(img, "1 c h w -> h w c")
            img_pil = Image.fromarray(img.type(torch.uint8).cpu().numpy())
            return_dict[f"output-{seed}"] = img_pil
            if args.visualize_all_seeds and seed == best_seed:
                return_dict[f"output-best_seed-{seed}"] = img_pil

        return_dict["output-mask"] = Image.fromarray(cv2.resize((self.mean_relevance_map * 255).astype(np.uint8), (512, 512)))

        return return_dict

    def accumulate_relevance_map(self, noise_diffs, current_step):
        if current_step < self.first_step_for_mask_extraction or current_step >= self.last_step_for_mask_extraction:
            return
        
        mean_diff = np.stack(noise_diffs, axis=0).mean(axis=0)  # (seed, H, W) → mean(H, W)
        masks = mean_diff ** 2  # sharpening mask
        masks = (masks - masks.min()) / (masks.max() - masks.min())  # normalizing to [0, 1]
        
        if self.mean_relevance_map is None:
            self.mean_relevance_map = masks
        else:
            self.mean_relevance_map = (self.mean_relevance_map * current_step + masks) / (current_step + 1)
    
    def calculate_BIS(self, input_image_np, z0_image):
        if isinstance(z0_image, torch.Tensor):
            z0_image_np = z0_image.detach().cpu().numpy()
        else:
            z0_image_np = np.array(z0_image)
            
        diff = np.abs(input_image_np - z0_image_np)
        diff = diff.mean(axis=1)[0]
        
        background_mask = cv2.resize(
            (1 - self.mean_relevance_map).astype(np.float32), (512, 512)
        )
        background_inconsistency_score = (diff * background_mask).sum() / background_mask.sum()
        return background_inconsistency_score
    
    def get_noise_diff(self, noise_cond, noise_uncond):
        diff = (noise_cond - noise_uncond).abs()[0].sum(dim=0).detach().cpu().numpy()

        # removing outliers
        Q1 = np.percentile(diff, 25, interpolation = 'midpoint') 
        Q3 = np.percentile(diff, 75, interpolation = 'midpoint') 
        IQR = Q3 - Q1
        factor = 1.5
        low_lim = Q1 - factor * IQR
        up_lim = Q3 + factor * IQR
        diff = np.clip(diff, 0, up_lim)

        # normalizing to [0, 1]
        diff = (diff - diff.min()) / (diff.max() - diff.min())

        return diff