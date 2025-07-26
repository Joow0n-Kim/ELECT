# **Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing (ELECT) (ICCV 2025)**

Official PyTorch implementation of  
[**“Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing”**](https://arxiv.org/abs/2504.13490) (ICCV 2025)

> #### Authors &emsp;&emsp; Joowon Kim<sup>1&#42;</sup>, Ziseok Lee<sup>2&#42;</sup>, Donghyeon Cho<sup>1</sup>, Sanghyun Jo<sup>3</sup>, Yeonsung Jung<sup>1</sup>, Kyungsu Kim<sup>1,2&dagger;</sup>, Eunho Yang<sup>1&dagger;</sup> <br> <sub> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup>1</sup>KAIST, <sup>2</sup>Seoul National University <sup>3</sup>OGQ</sub> <br> <sub> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup>&#42;</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding author</sub>


## 🚀 Introduction

<p align="center">
  <img src="./figures/ELECT_pipeline.jpg" width="720" alt="ELECT pipeline"/>
</p>

Despite impressive advances in diffusion models, **instruction-guided image editing** often fails—e.g. distorted backgrounds—due to the stochastic nature of sampling.  

**ELECT (Early-timestep Latent Evaluation for Candidate Selection)** tackles this by:

* **Multiple-seed baseline**: Uses a _Background Inconsistency Score (BIS)_ to reach Best-of-N performance _without supervision_.  
* **Zero-shot ranking**: Estimates background mismatch at **early diffusion timesteps**, selecting seeds that **keep the background while editing only the foreground**.  
* **Cost savings**: Cuts sampling **compute by 41 % on average (up to 61 %)**.  
* **Higher success**: Recovers ~40 % of previously failed edits, boosting background consistency & instruction adherence.  
* **Extensibility**: Integrates into instruction-guided pipelines _and_ MLLMs for joint **seed & prompt** selection when seed-only isn’t enough.

All benefits come **without additional training or external supervision**.



### ✨ Updates
| Date | Event |
|---|---|
 **2025-04-19** | 📚 arXiv pre-print released |
 **2025-06-26** | 🏆 Accepted to ICCV 2025 |
 **2025-07-26** | 💻 Initial code release |

## 🛠️ Get Started

1. Clone the repo

2. Create the environment. This is an example using conda.
```bash
conda create -n elect python=3.9
conda activate elect
pip install -r requirements.txt
```

3. (Optional) Use **InstructDiffusion**
- git clone https://github.com/cientgu/InstructDiffusion
- Download `v1-5-pruned-emaonly-adaption-task.ckpt` from that repo and move it to `./checkpoints`.

4. (Optional) Use **MGIE**
- Follow the [MGIE setup guide](https://github.com/apple/ml-mgie).  
- Place the official [LLaVA-Lightning-7B](https://huggingface.co/liuhaotian/LLaVA-Lightning-7B-delta-v1-1) in `./checkpoints/LLaVA-7B-v1`.  
- Put `mllm.pt` and `unet.pt` in `./checkpoints/mgie_7b`.

5. Datasets for evaluation

| Dataset | Link |
| --- | --- |
| **PIE-Bench** | <https://github.com/cure-lab/PnPInversion> |
| **MagicBrush test set** | <https://osu-nlp-group.github.io/MagicBrush/> |



## 🏃‍♂️ Run Experiments
### Run Baselines
- Single image
```
python inference.py \
  --run_type run_single_image \
  --input_path ./images/cat_to_bear.png \
  --instruction "Replace the cat with a bear" \
  --model {instructpix2pix | magicbrush | instructdiffusion | mgie | ultraedit}
```
- Dataset
```
python inference.py \
  --run_type run_dataset \
  --dataset_dir ./datasets/PIE-bench \
  --model {instructpix2pix | magicbrush | instructdiffusion | mgie | ultraedit}
```

### Run ELECT (seed selection)
```
python inference.py \
  --run_type run_single_image \
  --input_path ./images/cat_to_bear.png \
  --instruction "Replace the cat with a bear" \
  --model {instructpix2pix | magicbrush | instructdiffusion | mgie | ultraedit} \
  --select_one_seed \
  --num_random_candidates 10
```
- **Arguments**
    - _select_one_seed_: If set, select the best seed from the candidate seeds based on background inconsistency scores.
    - _num_random_candidates_: Number of random candidate seeds to be used for inference. (random seeds)
    - _candidate_seeds_: List of candidate seeds for fixed seed inference. (if num_random_candidates is 0, this will be used)
    - _stopping_step_: The step at which to select the best seed. (default=40) (assuming --inference_step 100; scale proportionally if you use a different total number of inference steps)
    - _first_step_for_mask_extraction_: The first step for relevance mask extraction. This is used to accumulate the relevance map. (default=0)
    - _last_step_for_mask_extraction_: The last step for relevance mask extraction. This is used to accumulate the relevance map. (default=20)
    - _visualize_all_seeds_: If set, visualize all seeds' outputs. Otherwise, only the best seed's output is visualized.

    - _output_dir_: Directory where edited images are saved (default: `./outputs`).
        - If **`visualize_all_seeds` is _not_ set**, the result is saved as  
          `{input_name}_output_{selected_seed}.png`.
        - If **`visualize_all_seeds` _is_ set**, every seed’s output is saved as  
          `{input_name}_output_{seed}.png`, and the elected one is additionally stored as  
          `{input_name}_output-best_seed-{seed}.png`.
        - The relevance map used for the Background Inconsistency Score is saved as  
          `{input_name}_output-mask.png`.
### Todos
- [x] Release implementation code for seed selection
- [ ] Release implementation code for prompt selection
- [ ] Release evaluation code

## 🙏 Acknowledgements
We gratefully acknowledge and extend our sincere appreciation to the creators of the following projects and datasets, whose excellent contributions laid the foundation for this work.

| Category | Repositories |
| --- | --- |
| **Models** | [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) · [InstructDiffusion](https://github.com/cientgu/InstructDiffusion) · [MGIE](https://github.com/apple/ml-mgie) · [UltraEdit](https://github.com/HaozheZhao/UltraEdit) |
| **Datasets** | [PIE-Bench](https://github.com/cure-lab/PnPInversion) · [MagicBrush](https://osu-nlp-group.github.io/MagicBrush/) |


## 📚 Citation
```bibtex
@article{kim2025early,
  title   = {Early timestep zero-shot candidate selection for instruction-guided image editing},
  author  = {Kim, Joowon and Lee, Ziseok and Cho, Donghyeon and Jo, Sanghyun and Jung, Yeonsung and Kim, Kyungsu and Yang, Eunho},
  journal = {arXiv preprint arXiv:2504.13490},
  year    = {2025}
}
```
