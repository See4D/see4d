#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader

from tqdm import tqdm
from einops import rearrange

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import transformers
from transformers import CLIPTokenizer

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import hydra
import wandb
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf

#private package
from mv_unet import SpatialTransformer3D
from mv_diffusion import mvdream_diffusion_model

# from src.dataset.data_module import worker_init_fn
from src.dataset.sampler import MultiNomialRandomSampler, ConcatDatasetWithIndex
from src.dataset.see4d_dataloader import get_combined_dataset, visualize_sample
from utils.train_utils import mask_pixels, read_train_imgs, prepare_extra_step_kwargs, worker_init_fn

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config.config import load_typed_root_config
    from src.config.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def init_mvd(args):
    single_view = args.single_view
    base_model_path = args.base_model_path

    if(single_view):
        mv_unet_path = base_model_path + "/unet/single/ema-checkpoint" if args.pretrain_unet is None else args.pretrain_unet
        print(mv_unet_path)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    else:
        mv_unet_path = base_model_path + "/unet/sparse/ema-checkpoint" if args.pretrain_unet is None else args.pretrain_unet
        print(mv_unet_path)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")

    rgb_model = mvdream_diffusion_model(base_model_path,mv_unet_path,tokenizer,seed=12345)
    # mv_net_path = base_model_path + "/unet/SR/ema-checkpoint"
    # rgb_model_SR = mvdream_diffusion_model_SR(base_model_path,mv_unet_path,tokenizer,quantization=False,seed=12345)
    return rgb_model, None 

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)

def main(cfg_dict: DictConfig):
    
    # Load the configuration
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    args = cfg.diffusion
    cfg.dataset.image_shape = [args.height, args.width]
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.output_dir = os.path.join(args.output_dir, cfg_dict.wandb.name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,#args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dispatch_batches=True,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    ## Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_path, subfolder="scheduler") 
    # Potentially load in the weights and states from a previous save
    ckpt_path = None

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            ckpt_path = os.path.basename(args.resume_from_checkpoint)
        elif os.path.exists(args.output_dir):
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            ckpt_path = dirs[-1] if len(dirs) > 0 else None

        ## Over-write pretrain unet to load unet parameters
        ## if not running on existing model, then keep the pretrain_unet so that we can load pre-trained single view model
        if ckpt_path is not None:
            args.pretrain_unet = os.path.join(args.output_dir, ckpt_path)
        print(f"Resuming from checkpoint: {args.pretrain_unet}")

    rgb_model, rgb_model_SR = init_mvd(args)

    if not args.train_super_resolution:
        unet = rgb_model.unet.to(accelerator.device)
        vae = rgb_model.pipe.vae.to(accelerator.device)
        rgb_model.pipe.text_encoder = rgb_model.pipe.text_encoder.to(accelerator.device)

        encode_prompt = rgb_model.pipe._encode_prompt
        encode_image = rgb_model.pipe.encode_image

        get_wt = rgb_model.custom_decay_function_weight
        scaling = rgb_model.pipe.vae.config.scaling_factor
    else:
        unet = rgb_model_SR.unet.to(accelerator.device)
        vae = rgb_model_SR.pipe.vae.to(accelerator.device)

        encode_prompt = rgb_model_SR.pipe._encode_prompt
        encode_image = rgb_model_SR.pipe.encode_image
        get_wt = rgb_model_SR.custom_decay_function_weight
        scaling = rgb_model_SR.pipe.vae.config.scaling_factor 

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    unet.requires_grad_(True)

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open(os.path.join(args.output_dir, 'rec_para.txt'), 'w')
        rec_txt2 = open(os.path.join(args.output_dir, 'rec_para_train.txt'), 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    # ipdb.set_trace()
    step_tracker = StepTracker() ## step tracker is used to adjust view sampler

    # Create combined training dataset
    combined_train = get_combined_dataset(args.dataset_root, split="train")
    sampler = MultiNomialRandomSampler(combined_train, p=args.datasets_cfg.prob, main_process=accelerator.is_main_process)
    train_dataset = ConcatDatasetWithIndex(combined_train)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size, #cfg.data_loader.train.batch_size,
        shuffle=False, # randomsampler or IterableDataset do not need shuffle
        num_workers=args.num_workers,
    )

    dataset_list = ['Syncam4D', 'Kubric4D', 'Obj4D10k', 'recammaster']
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
       len(train_dataloader) / (args.gradient_accumulation_steps*accelerator.num_processes))
    # num_update_steps_per_epoch = 1000
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    #lr_scheduler is constant
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        if args.report_to == "wandb":
            output_dir = Path(args.output_dir)
            wandb_extra_kwargs = {'id': f"{output_dir.name}" if cfg_dict.wandb.id is None else cfg_dict.wandb.id, 'resume': "allow"} ## previously is must here
            wandb_extra_kwargs.update({
                "entity": cfg_dict.wandb.entity,
                # "mode": "disabled",#cfg_dict.wandb.mode,
                "mode": cfg_dict.wandb.mode,
                "name": f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
                "tags": cfg_dict.wandb.get("tags", None),
                # "log_model":False,
                "dir": args.output_dir,
                "config":OmegaConf.to_container(cfg_dict),
                # **wandb_extra_kwargs,
            })
            accelerator.init_trackers("See4D", init_kwargs={"wandb": wandb_extra_kwargs})
        else:
            accelerator.init_trackers("See4D", config=None)

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    resume_step = 0
    if ckpt_path is not None:
        accelerator.print(f"Resuming from checkpoint {ckpt_path}")
        global_step = int(ckpt_path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * args.gradient_accumulation_steps)
            
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    inference_times = 0
    step_tracker.set_step(global_step)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):     

            step_tracker.set_step(global_step)
            
            with accelerator.accumulate(unet):

                #input video is 0 to 1
                input = batch['video'].to(accelerator.device, dtype = weight_dtype)
                target_mask = batch['target_mask'].to(accelerator.device, dtype = weight_dtype)
                
                batch_size, num_frames, _, height, width = input.shape

                condition_masks = torch.ones_like(target_mask).to(accelerator.device, dtype = weight_dtype)
                masks = torch.concat([condition_masks, target_mask], dim=1)

                input = rearrange(input, "b f c h w -> (b f) c h w", f=num_frames)
                masks = rearrange(masks, "b f c h w -> (b f) c h w", f=num_frames)
                warp = input * masks

                # warp, masks = mask_pixels(input)#-1 to 1 

                # # sanity check for warp image
                # if accelerator.is_main_process:
                #     sample = (input[:num_frames]).cpu()
                #     warp_sample = (warp[:num_frames]).cpu()
                #     os.makedirs(os.path.join(args.output_dir, f"sanity-check"), exist_ok=True)
                #     save_path = os.path.join(args.output_dir, f"sanity-check/step-{global_step}-warp.png")
                #     visualize_sample(warp_sample.float(), save_path, None, None)
                #     save_path = os.path.join(args.output_dir, f"sanity-check/step-{global_step}.png")
                #     visualize_sample(sample.float(), save_path, None, None)
                # global_step+=1
                # continue

                input = input * 2 - 1
                warp = warp * 2 - 1
                # vae input should be -1 to 1
                input_latents = vae.encode(input).latent_dist.sample() * scaling
                warp_latents = vae.encode(warp).latent_dist.sample() * scaling

                input_mask = torch.ones_like(input).mean(dim=1, keepdim=True)

                mask_latents = torch.nn.functional.interpolate(
                    masks,
                    size=(
                        height // 8,
                        width // 8
                    )
                ).to(weight_dtype).to(accelerator.device)# [b*f, c, h, w]

                input_mask = torch.nn.functional.interpolate(
                    input_mask,
                    size=(
                        height // 8,
                        width // 8
                    )
                ).to(weight_dtype).to(accelerator.device)# [b*f, c, h, w]

                # context_latents = rearrange(context_latents, "(b f) c h w -> b f c h w", f=num_frames)
                warp_latents = rearrange(warp_latents, "(b f) c h w -> b f c h w", f=num_frames)
                input_latents = rearrange(input_latents, "(b f) c h w -> b f c h w", f=num_frames)

                input_mask = rearrange(input_mask, "(b f) c h w -> b f c h w", f=num_frames)
                mask_latents = rearrange(mask_latents, "(b f) c h w -> b f c h w", f=num_frames)

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device)
                timesteps = timesteps.long()
                # timesteps = 999*torch.ones_like(timesteps)

                timestep_warp = (timesteps//5).long()
                w_t = get_wt(timestep_warp.float())
                w_t = w_t.view(w_t.shape[0], 1, 1, 1, 1).to(weight_dtype)

                noise = torch.randn_like(input_latents)
                noise_warp = torch.randn_like(input_latents)
                warp_noisy_latents = noise_scheduler.add_noise(warp_latents, noise_warp, timestep_warp)
                input_noisy_latents = noise_scheduler.add_noise(input_latents, noise, timesteps)
                warp_noisy_latents = w_t * warp_noisy_latents + (1 - w_t) * input_noisy_latents

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(input_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                context_mask = torch.zeros_like(input_mask).to(accelerator.device)
                context_mask[:, :num_frames//2] = 1

                if args.conditioning_dropout_prob is not None and random.random() < args.conditioning_dropout_prob:
                    warp_noisy_latents = torch.zeros_like(warp_noisy_latents)
                    mask_latents_zero = torch.zeros_like(mask_latents)
                    mask_latents = mask_latents_zero * (1 - context_mask) + mask_latents * context_mask

                input_noisy_latents = input_noisy_latents*(1-context_mask) + input_latents*context_mask
                warp_noisy_latents = warp_noisy_latents*(1-context_mask) + input_latents*context_mask
                mask_latents = mask_latents*(1-context_mask) + input_mask*context_mask

                latent_model_input = torch.cat([input_noisy_latents, warp_noisy_latents, mask_latents], dim=2) # b,f,c+c_img+c_ray+1,h,w
                
                #diffusion condition
                with torch.no_grad():
                    prompt_embeds = encode_prompt('', accelerator.device, num_frames, False, None)
                    prompt_embeds = prompt_embeds.unsqueeze(0).repeat(batch_size,1,1,1)#1,16,77,1024

                    #encoder input should -1 to 1
                    image_prompt  = rearrange(input, "(b f) c h w -> b f c h w", f=num_frames)
                    image_prompt = image_prompt[:,:num_frames//2]
                    image_prompt  = rearrange(image_prompt, "b f c h w -> (b f) c h w", f=num_frames//2)
                    _, image_embeds_pos = encode_image(image_prompt, accelerator.device, num_frames//2)#8,77,1024
                    image_embeds_pos = rearrange(image_embeds_pos, "(b f) l w -> b f l w", f=num_frames//2)
                    image_embeds_pos = image_embeds_pos.repeat(1,2,1,1)

                latent_model_input = latent_model_input.reshape([batch_size*num_frames, 9, 
                                                                    input_latents.shape[-2], input_latents.shape[-1]])

                condition_embeds = prompt_embeds + image_embeds_pos
                condition_embeds = rearrange(condition_embeds, "b f l w -> (b f) l w", f=num_frames)#.detach()
                condition_embeds = condition_embeds.to(dtype=weight_dtype)

                timesteps = timesteps.repeat_interleave(num_frames).to(latent_model_input.dtype)
                unet_inputs = {
                    'x': latent_model_input.detach(),# torch.Size([num_frames, 5, 32, 32])
                    'timesteps': timesteps,
                    'context': condition_embeds.detach(),#.to(unet.device),
                    'num_frames': num_frames,# 4
                    'camera': None,#
                }
                model_pred = unet.forward(**unet_inputs)# [6, 4, 32, 32]
                model_pred = model_pred.reshape([batch_size, num_frames, -1, input_latents.shape[-2], input_latents.shape[-1]])#[:,:,:4]

                # if global_step % 1 == 0:
                        
                #     extra_step_kwargs = prepare_extra_step_kwargs(None, 0.0, noise_scheduler)
                #     pred_latents = noise_scheduler.step(
                #             model_pred[0,:,:4], timesteps, latent_model_input[:, :4], **extra_step_kwargs, return_dict=False
                #         )[1]
                
                #     pred_latents = 1 / vae.config.scaling_factor * pred_latents
                #     image = vae.decode(pred_latents.to(vae.dtype)).sample
                #     image = (image / 2 + 0.5).clamp(0, 1)
                #     image = image*255.0
                #     # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                #     image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
                #     image = image.astype(np.uint8)
                #     imageio.mimwrite(f'visualization/test_output_{step}_{timesteps.item()}.mp4', list(image))

                loss = F.mse_loss(model_pred.float()[:,num_frames//2:,:4], target.float()[:,num_frames//2:], reduction="mean") #+ 0.0 * model_pred.float()[:,:,4:].mean()
                loss += 0.0 * sum(torch.norm(param, p=2) ** 2 for param in unet.parameters())

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    #accelerator.unscale_gradients()
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    }, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        accelerator.unwrap_model(unet).save_pretrained(save_path)
                        logger.info(f"Saved state to {save_path}")
                        if global_step % int(10*args.checkpointing_steps) == 0:
                            dest = save_path.replace("checkpoint", "ckpt")
                            shutil.copytree(save_path, dest, dirs_exist_ok=True)
                            
                    # sample images!
                    if (
                        (global_step % args.validation_steps == 0)
                    ):
                    
                    # if True:
                        logger.info(
                            f"Running validation... Step {global_step}."
                        )
                        inference_times += 1

                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        unet.eval()
                        rgb_model.unet = unet

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ):  
                            
                            combined_test = get_combined_dataset(args.dataset_root, split="test")

                            fixed_seed = 42
                            eval_generator = torch.Generator()
                            eval_generator.manual_seed(fixed_seed)

                            for dataset_index, evaluation_dataset in enumerate(combined_test):

                                eval_sampler = RandomSampler(evaluation_dataset, generator=eval_generator)
                                evaluation_dataloader = DataLoader(
                                    evaluation_dataset,
                                    sampler=eval_sampler,
                                    batch_size=1, #cfg.data_loader.train.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    worker_init_fn=worker_init_fn
                                )

                                eval_batch = next(iter(evaluation_dataloader))

                                input = eval_batch['video'].to(accelerator.device, dtype = weight_dtype)
                                target_mask = eval_batch['target_mask'].to(accelerator.device, dtype = weight_dtype)

                                batch_size, num_frames, _, height, width = input.shape

                                condition_masks = torch.ones_like(target_mask).to(accelerator.device, dtype = weight_dtype)
                                masks = torch.concat([condition_masks, target_mask], dim=1)

                                input = rearrange(input, "b f c h w -> (b f) c h w", f=num_frames)
                                masks = rearrange(masks, "b f c h w -> (b f) c h w", f=num_frames)
                                warp = input * masks

                                # warp, masks = mask_pixels(input)#-1 to 1 

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
                                images_predict_batch = rgb_model.inference_next_frame(prompt,batch,num_frames,
                                                                                        height,width,gt_num,output_type='pil')
                                
                                dataset_name = dataset_list[dataset_index]
                                sample = ((input+1.0)/2.0).cpu()                
                                warp_sample = ((warp+1.0)/2.0).cpu()

                                save_path = os.path.join(val_save_dir, f"{dataset_name}-validation-{global_step}-see4d.png")
                                visualize_sample(sample.float(), save_path, 
                                                    None, None, warp_sample.float(), images_predict_batch)

                        torch.cuda.empty_cache()
                        unet.train()

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        accelerator.unwrap_model(unet).save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
