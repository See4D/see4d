import os
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import RandomSampler, DataLoader
from einops import rearrange
from accelerate import Accelerator
from transformers import CLIPTokenizer
from tqdm import tqdm
import imageio

from mv_diffusion import mvdream_diffusion_model
from src.dataset.iphone import iPhoneDataset
from utils.train_utils import worker_init_fn
from src.evaluation.metrics import (
    compute_psnr_np,
    compute_ssim_np,
    compute_lpips_np,
)


def save_video(save_path, sample):

    sample = sample*255.0
    sample = sample.float().numpy() if isinstance(sample, torch.Tensor) else sample
    sample = sample.astype(np.uint8)
    imageio.mimwrite(save_path, list(sample))

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on iPhoneDataset")
    p.add_argument(
        "--base_model_path",
        type=str,
        default='checkpoint/MVD_weights',
        help="Path to the pretrained Diffusers model (tokenizer, scheduler, etc.)",
    )
    p.add_argument(
        "--mv_unet_path",
        type=str,
        default='outputs/4d_train_withrecam/ckpt-10000',
        help="Directory where your U-Net was saved (e.g. output_dir/checkpoint-*)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers",
    )
    return p.parse_args()


def main():
        
    args = parse_args()
    
    base_model_path = args.base_model_path
    mv_unet_path = args.mv_unet_path

    video_save_dir = os.path.join(mv_unet_path, 'pred_video')
    os.makedirs(video_save_dir, exist_ok=True)

    weight_dtype = torch.float16

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    #load ckpt
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    rgb_model = mvdream_diffusion_model(base_model_path,
                                        mv_unet_path,tokenizer,seed=12345)

    rgb_model.unet.to(device)
    rgb_model.pipe.vae.to(device)
    rgb_model.pipe.text_encoder.to(device)
    # encode_prompt = rgb_model.pipe._encode_prompt
    # encode_image = rgb_model.pipe.encode_image
    # get_wt = rgb_model.custom_decay_function_weight
    # scaling = rgb_model.pipe.vae.config.scaling_factor


    test_dataset = iPhoneDataset()

    fixed_seed = 42
    eval_generator = torch.Generator()
    eval_generator.manual_seed(fixed_seed)

    results = {"psnr": [], "ssim": [], "lpips": []}
    eval_sampler = RandomSampler(test_dataset, generator=eval_generator)

    evaluation_dataloader = DataLoader(
        test_dataset,
        sampler=eval_sampler,
        batch_size=1, 
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn
    )

    for idx, eval_batch in enumerate(tqdm(evaluation_dataloader)):

        input = eval_batch['video'].to(accelerator.device, dtype = weight_dtype)
        target_mask = eval_batch['target_mask'].to(accelerator.device, dtype = weight_dtype)
        target = eval_batch['target'].to(accelerator.device, dtype = weight_dtype)

        
        batch_size, num_frames, _, height, width = input.shape

        input = torch.concat([input[:,:num_frames//2], target], dim=1)

        condition_masks = torch.ones_like(target_mask).to(accelerator.device, dtype = weight_dtype)
        masks = torch.concat([condition_masks, target_mask], dim=1)

        input = rearrange(input, "b f c h w -> (b f) c h w", f=num_frames)
        masks = rearrange(masks, "b f c h w -> (b f) c h w", f=num_frames)
        warp = input * masks

        # if accelerator.is_main_process:
        #     sample = (input[:num_frames]).cpu()
        #     warp_sample = (warp[:num_frames]).cpu()
        #     os.makedirs(os.path.join(args.output_dir, f"sanity-check"), exist_ok=True)
        #     save_path = os.path.join(args.output_dir, f"sanity-check/step-test-{global_step}-warp.png")
        #     visualize_sample(warp_sample.float(), save_path, None, None)
        #     save_path = os.path.join(args.output_dir, f"sanity-check/step-test-{global_step}.png")
        #     visualize_sample(sample.float(), save_path, None, None)

        input = input * 2 - 1
        warp = warp * 2 - 1

        input_masks = torch.ones(batch_size, num_frames, 1, height, width).to(accelerator.device, 
                                                                                dtype = weight_dtype)

        gt_num = num_frames//2
        context_mask = torch.zeros_like(input_masks).to(accelerator.device)
        context_mask[:, :gt_num] = 1

        input_masks = rearrange(input_masks, "b f c h w -> (b f) c h w", f=num_frames)
        context_mask = rearrange(context_mask, "b f c h w -> (b f) c h w", f=num_frames)

        original_input = input.clone()
        condition_pixel_values = warp*(1-context_mask) + input*context_mask
        masks_pixel_values = masks*(1-context_mask) + input_masks*context_mask

        batch = {
                'conditioning_pixel_values': condition_pixel_values,
                'masks': masks_pixel_values,
                'input': original_input,
        }

        prompt = ['']
        with torch.autocast(device.type, enabled=True):
            images_predict_batch = rgb_model.inference_next_frame(prompt,batch,num_frames,
                                                                height,width,gt_num,output_type='pil')
        
        dataset_name = 'iphone'
        sample = ((input+1.0)/2.0).cpu()
        sample = torch.concat((sample[:gt_num], target[0].cpu()), dim = 0)
        warp_sample = ((warp+1.0)/2.0).cpu()
        
        #evaluation
        pred_imgs = np.array(images_predict_batch[gt_num:]).transpose(0,3,1,2)
        gt_imgs = np.array(target[0].cpu())

        psnr = compute_psnr_np( 
            ground_truth=gt_imgs,
            predicted=pred_imgs,
        ).mean()

        ssim = compute_ssim_np(
            ground_truth=gt_imgs,
            predicted=pred_imgs,
        ).mean()

        lpips = compute_lpips_np(
            ground_truth=gt_imgs,
            predicted=pred_imgs,
            device = accelerator.device,
        ).mean()

        print(f"{dataset_name} sample: PSNR {psnr}")
        print(f"{dataset_name} sample: LPIPS {lpips}")
        print(f"{dataset_name} sample: SSIM {ssim}")
        results["psnr"].append(psnr)
        results["ssim"].append(ssim)
        results["lpips"].append(lpips)
        

        source_save_dir = os.path.join(video_save_dir, f"iphone-video-{idx}-source.mp4")
        save_video(source_save_dir, sample[:num_frames//2].permute(0, 2, 3, 1))

        target_save_dir = os.path.join(video_save_dir, f"iphone-video-{idx}-target.mp4")
        save_video(target_save_dir, sample[num_frames//2:].permute(0, 2, 3, 1))

        warp_save_dir = os.path.join(video_save_dir, f"iphone-video-{idx}-warp.mp4")
        save_video(warp_save_dir, warp_sample[num_frames//2:].permute(0, 2, 3, 1))

        predict_save_dir = os.path.join(video_save_dir, f"iphone-video-{idx}-predict.mp4")
        save_video(predict_save_dir, images_predict_batch[num_frames//2:])
        

    print("\n==== iPhoneDataset Evaluation ====")
    print(f"Average PSNR  : {np.mean(results['psnr']):.4f}")
    print(f"Average SSIM  : {np.mean(results['ssim']):.4f}")
    print(f"Average LPIPS : {np.mean(results['lpips']):.4f}")

if __name__ == "__main__":
    main()