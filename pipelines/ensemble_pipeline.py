import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils.torch_utils import randn_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

class StableDiffusionInstructPix2PixEnsemblePipeline (StableDiffusionInstructPix2PixPipeline):
    """
    StableDiffusionInstructPix2PixEnsemblePipeline
    This class is a custom pipeline for ensemble inference using Stable Diffusion InstructPix2Pix.
    It is reconstructed based on the `__call__` function of `StableDiffusionInstructPix2PixPipeline`.
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
        prompt=None,
        image=None,
        num_inference_steps=100,
        stopping_step=40,
        first_step_for_mask_extraction=0,
        last_step_for_mask_extraction=20,
        guidance_scale=7.5,
        image_guidance_scale=1.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        **kwargs,
    ):
        """
        This function is reconstructed based on the __call__ method of StableDiffusionInstructPix2PixPipeline.
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
                image latents as `image`, but if passing latents directly it is not encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
                
            stopping_step (`int`, *optional*, defaults to 40): 
                *Argument for ELECT*. The step at which to select the best seed.
            first_step_for_mask_extraction (`int`, *optional*, defaults to 0):
                *Argument for ELECT*. The first step for relevance mask extraction. This is used to accumulate the relevance map.
            last_step_for_mask_extraction (`int`, *optional*, defaults to 20):
                *Argument for ELECT*. The last step for relevance mask extraction. This is used to accumulate the relevance map.
                
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Push the generated image towards the initial `image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            **kwargs: Additional keyword arguments, including 'args' for configuration.
        Returns:
            dict: A dictionary containing:
                - "output-{seed}": PIL.Image for each seed's output. (if args.visualize_all_seeds is False, only the best seed is returned)
                - "output-best_seed": PIL.Image for the best seed (if args.visualize_all_seeds is True).
                - "output_mask": PIL.Image of the mean relevance map.
        """
        
        args = kwargs["args"]
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        device = self._execution_device
        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        source_image_pil = image
        image = self.image_processor.preprocess(image)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

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
        num_channels_latents = self.vae.config.latent_channels
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        added_cond_kwargs = None

        decoded_input = self.vae.decode(image_latents.chunk(3)[0], return_dict=False)[0]
        input_image_np = decoded_input.detach().cpu().numpy()

        candidate_seeds = args.candidate_seeds
        active_seeds = list(candidate_seeds)
        num_seeds = len(candidate_seeds)

        seed_latents_dict = {}
        for idx, seed in enumerate(candidate_seeds):
            tmp_generator = torch.Generator(device="cuda").manual_seed(seed)
            initial_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                tmp_generator,
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

        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                eps_diffs = [] # noise differences for each seed
                for seed in active_seeds:
                    tmp_dict = seed_latents_dict[seed]
                    tmp_latents = tmp_dict["latents"]
                    tmp_generator = tmp_dict["generator"]

                    latent_model_input = torch.cat([tmp_latents] * 3)
                    scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                    noise_pred = self.unet(
                        scaled_latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        cross_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                    # step function
                    self.scheduler._init_step_index(t)
                    sigma = self.scheduler.sigmas[self.scheduler.step_index].to(torch.float16)
                    z0t = tmp_latents - sigma * noise_pred
                    sigma_from = self.scheduler.sigmas[self.scheduler.step_index]
                    sigma_to = self.scheduler.sigmas[self.scheduler.step_index + 1]
                    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
                    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
                    derivative = (tmp_latents - z0t) / sigma
                    dt = sigma_down - sigma
                    tmp_latents = tmp_latents + derivative * dt
                    noise = randn_tensor(noise_pred.shape, dtype=noise_pred.dtype, device=noise_pred.device, generator=tmp_generator)
                    tmp_latents = tmp_latents + noise * sigma_up

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