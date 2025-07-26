import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pipelines.stable_diffusion_3_pipeline import StableDiffusion3InstructPix2PixPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

class StableDiffusion3InstructPix2PixEnsemblePipeline (StableDiffusion3InstructPix2PixPipeline):
    """
    StableDiffusion3InstructPix2PixEnsemblePipeline
    This class is a custom pipeline for ensemble inference using Stable Diffusion3 InstructPix2Pix (from UltraEdit repository).
    It is reconstructed based on the `__call__` function of `StableDiffusion3InstructPix2PixPipeline`.
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
    def get_best_output_from_various_seeds(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        image = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 100,
        stopping_step: int = 40,
        first_step_for_mask_extraction: int = 0,
        last_step_for_mask_extraction: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        mask_img = None,
        **kwargs
    ):
        
        args = kwargs['args']
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
        )
        
        if self.do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds], dim=0)
            
            # Similiarly
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, negative_pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0)          
            
        # 3. Preprocess image
        image = self.image_processor.preprocess(image)
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare Image latent
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor
        
        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels

        # 7. Check that shapes of latents and image match the DIT in_channels
        num_channels_image = image_latents.shape[1]
        if mask_img is not None:
            mask_img = self.image_processor.preprocess(mask_img)
            mask_image_latents = self.prepare_image_latents(
                mask_img,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                self.do_classifier_free_guidance,
            )
            num_channels_image += mask_image_latents.shape[1]

        if num_channels_latents + num_channels_image != self.transformer.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.transformer`: {self.transformer.config} expects"
                f" {self.transformer.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.transformer` or your `image` input."
            )
            
            
        # 8. Prepare ELECT pipeline
        decoded_input = self.vae.decode(image_latents.chunk(3)[0], return_dict=False)[0]
        input_image_np = decoded_input.detach().cpu().numpy()
        
        candidate_seeds = args.candidate_seeds
        active_seeds = list(candidate_seeds)
        num_seeds = len(candidate_seeds)
        
        seed_latents_dict = {}
        for idx, seed in enumerate(candidate_seeds):
            tmp_generator = torch.Generator(device='cuda').manual_seed(seed)
            initial_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                tmp_generator,
                None,
            )
            seed_latents_dict[seed] = {
                "latents": initial_latents,
                "generator": tmp_generator,
                "idx": idx
            }
        
        
        self.mean_relevance_map = None
        self.first_step_for_mask_extraction = first_step_for_mask_extraction
        self.last_step_for_mask_extraction = last_step_for_mask_extraction

        current_z0ts = {} # z0t (predicted x0) for each seed at stopping step
        last_outputs = {}
        last_z0s = {}

        # 9. Denoising loop
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                eps_diffs = [] # noise differences for each seed
                for seed in active_seeds:
                    tmp_dict = seed_latents_dict[seed]
                    tmp_latents = tmp_dict["latents"]
                    tmp_generator = tmp_dict["generator"]
                    
                    latent_model_input = torch.cat([tmp_latents] * 3)
                    timestep = t.expand(latent_model_input.shape[0])
                    scaled_latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
                    if mask_img is not None:
                        scaled_latent_model_input = torch.cat([scaled_latent_model_input, mask_image_latents], dim=1)

                    noise_pred = self.transformer(
                        hidden_states=scaled_latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        # mask_index= mask_index,
                    )[0]

                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                    # step function
                    self.scheduler._init_step_index(t)
                    sigma = self.scheduler.sigmas[self.scheduler.step_index]
                    sigma_next = self.scheduler.sigmas[self.scheduler.step_index + 1]
                    z0t = tmp_latents - sigma * noise_pred
                    prev_sample = tmp_latents + (sigma_next - sigma) * noise_pred
                    tmp_latents = prev_sample.to(tmp_latents.dtype)
                    self.scheduler._step_index += 1

                    eps_diffs.append(self.get_noise_diff(noise_pred_text, noise_pred_image))
                    
                    seed_latents_dict[seed]["latents"] = tmp_latents # update latents of each seed
                
                    if i == stopping_step - 1:
                        current_z0ts[seed] = z0t
                        
                self.accumulate_relevance_map(eps_diffs, current_step = i)

                # selecting best seed at stopping step
                if i == stopping_step - 1:
                    background_inconcistency_scores = {}
                    for seed in candidate_seeds:
                        z0_image = self.vae.decode(current_z0ts[seed] / self.vae.config.scaling_factor, return_dict=False)[0]
                        background_inconcistency_scores[seed] = self.calculate_BIS(input_image_np, z0_image)
                    best_seed, best_score = min(background_inconcistency_scores.items(), key=lambda x: x[1])
                            
                    # After selecting the best seed, remove the rest
                    # If visualize_all_seeds is True, keep all seeds for visualization
                    if not args.visualize_all_seeds:
                        active_seeds = [best_seed]
            
            for seed in active_seeds:
                latents = seed_latents_dict[seed]["latents"]
                image_tensor = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                last_z0s[seed] = image_tensor
        
        
        return_dict = {}

        for seed, img in last_z0s.items():
            img_pil = self.image_processor.postprocess(
                img.detach(), output_type="pil", do_denormalize=[True]
            )[0]
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