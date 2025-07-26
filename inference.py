import os
import json
from argparse import ArgumentParser
import torch
from PIL import Image
from tqdm import tqdm
import random

from utils import load_model, image_resize, rle_decode

def generate (args, pipe, input_image, instruction, seed):
    # Call original __call__ functions to generate images
    # This function is used to generate images for a single image inference
    res = {}
    
    if args.model == "instructpix2pix" or args.model == "magicbrush":
        edited_image = pipe(
            image=input_image, 
            prompt=[instruction], 
            generator=torch.Generator(device='cuda').manual_seed(seed), 
            guidance_scale=args.text_cfg_scale, 
            image_guidance_scale=args.image_cfg_scale, 
            num_inference_steps=args.inference_steps, 
            args=args
        )
        res['output'] = edited_image.images[0]
        
    elif args.model == 'instructdiffusion':
        res = pipe(
            args=args, 
            prompt=instruction,
            input_image=input_image
        )
        
    elif args.model == "mgie":
        prompt_embeds, negative_prompt_embeds = pipe.generate_prompt(input_image, instruction)
        edited_image = pipe(
            image=input_image, 
            prompt_embeds=prompt_embeds, 
            negative_prompt_embeds=negative_prompt_embeds, 
            generator=torch.Generator(device='cuda').manual_seed(seed), 
            args=args
        )
        res['output'] = edited_image.images[0]
        
    elif args.model == 'ultraedit': 
        edited_image  = pipe(
            image=input_image, 
            prompt=instruction, 
            height=args.resolution, 
            width=args.resolution, 
            generator=torch.Generator(device='cuda').manual_seed(seed), 
            guidance_scale=args.text_cfg_scale, 
            image_guidance_scale=args.image_cfg_scale, 
            num_inference_steps=args.inference_steps
        )
        res['output'] = edited_image.images[0]
        
    else:
        raise ValueError(f"Model {args.model} not supported.")
    
    return res

def main():
    parser = ArgumentParser()
    
    parser.add_argument("--run_type", default="run_single_image", type=str, 
                                    choices=["run_single_image","run_dataset"])
    parser.add_argument("--model", default="instructpix2pix", type=str, 
                                   choices=["instructpix2pix", "magicbrush", "mgie", "instructdiffusion", "ultraedit"])
    parser.add_argument("--output_dir", default="./output", type=str, help="Directory to save the output images.")
    parser.add_argument("--output_name", default='test', type=str, help="Name of the output images.")
    
    # for single image inference
    parser.add_argument("--input_path", default=None, type=str, help="Path to the input image for single image inference.")
    parser.add_argument("--instruction", default=None, type=str, help="Instruction for single image inference.")
    
    # for dataset inference
    parser.add_argument("--dataset_dir", default="./datasets/PIE-bench", type=str)
    
    # default parameters for instruction-guided image editing models
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--inference_steps", default=100, type=int)
    parser.add_argument("--text_cfg_scale", default=7.5, type=float)
    parser.add_argument("--image_cfg_scale", default=1.5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    
    # arguments for ELECT
    parser.add_argument("--select_one_seed", action='store_true', help="If set, select the best seed from the candidate seeds based on background inconsistency scores.")
    parser.add_argument("--num_random_candidates", default=0, type=int, help="Number of random candidate seeds to be used for inference.")
    parser.add_argument("--candidate_seeds", nargs="+", default=[1], type=int, help="List of candidate seeds for fixed seed inference. (if num_random_candidates is 0, this will be used)")
    parser.add_argument("--stopping_step", default=40, type=int, help="The step at which to select the best seed.")
    parser.add_argument("--first_step_for_mask_extraction", default=0, type=int,
                                    help="The first step for relevance mask extraction. This is used to accumulate the relevance map.")
    parser.add_argument("--last_step_for_mask_extraction", default=20, type=int,
                                    help="The last step for relevance mask extraction. This is used to accumulate the relevance map.")
    
    parser.add_argument("--visualize_all_seeds", action='store_true', help="If set, visualize all seeds' outputs. Otherwise, only the best seed's output is visualized.")
    
    args = parser.parse_args()
    
    pipe = load_model(args)
    
    args.output_dir = os.path.join(args.output_dir, f"{args.model}-{args.output_name}")
    os.makedirs(args.output_dir, exist_ok = True)
    
    if args.run_type == "run_single_image":
        assert args.input_path is not None, "Input path must be provided for single image inference."
        assert args.instruction is not None, "Instruction must be provided for single image inference."
        input_name = os.path.basename(args.input_path).split('.')[0]
        data = {
            input_name: {
                "image_path": args.input_path,
                "editing_instruction": args.instruction
            }
        }
    elif args.run_type == "run_dataset":
        assert args.dataset_dir is not None, "Dataset directory must be provided for dataset inference."
        if "PIE" in args.dataset_dir:
            data = json.load(open(os.path.join(args.dataset_dir, 'mapping_file.json'), 'r'))
            args.datatype = "PIE"
        elif "magicbrush" in args.dataset_dir:
            data = json.load(open(os.path.join(args.dataset_dir, 'preprocessed.json'), 'r'))
            args.datatype = "magicbrush"
    else:
        raise ValueError(f"Run type {args.run_type} not supported.")
    
    if args.num_random_candidates > 0:
        args.candidate_seeds = random.sample(range(1, 100000), args.num_random_candidates)

    for img_key, item in data.items():
        image = Image.open(item['image_path']).convert('RGB')
        image = image_resize(image, resolution=args.resolution)
        instruction = item['editing_instruction']
        
        if args.select_one_seed:
            res = pipe.get_best_output_from_various_seeds(
                prompt=[instruction], 
                image=image, 
                num_inference_steps=args.inference_steps, 
                stopping_step=args.stopping_step,
                first_step_for_mask_extraction=args.first_step_for_mask_extraction,
                last_step_for_mask_extraction=args.last_step_for_mask_extraction,
                args=args
            )
            
            image.save (os.path.join(args.output_dir, f'{img_key}-input.png'))
            for key in res.keys():
                res[key].save (os.path.join(args.output_dir, f'{img_key}-{key}.png'))
            
        else: # output all candidate seeds
            for seed in args.candidate_seeds:
                args.seed = seed
                print(f"Processing {img_key} with seed {seed}...")
                
                res = generate(args, pipe, image, instruction, seed)          

                if not os.path.exists(os.path.join(args.output_dir, f'{img_key}-input.png')):
                    image.save (os.path.join(args.output_dir, f'{img_key}-input.png'))
                
                for key in res.keys():
                    res[key].save (os.path.join(args.output_dir, f'{img_key}-{key}-seed{seed}.png'))

                
if __name__ == "__main__":
    main()
