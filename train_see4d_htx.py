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

"""Script to fine-tune Stable Video Diffusion.
Add different noise for latents
Remove cross attention
"""
import argparse
import random
import logging
import math
import os
import glob
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange, repeat

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers import DDPMScheduler

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from torch.utils.data import Dataset

from pathlib import Path

import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

#from diffusers.models import UNetSpatioTemporalConditionModel as SVDUnet
#
#from diffusion.pipelines import create_pipeline
#from diffusion.models.unet_time_4d import UNetTime4DModel as STUnet
#from diffusion.utils.cam_utils import get_relative_pose_batch, get_rays
#from diffusion.utils.common_utils import save_image_video
#from diffusion.utils.train_utils import tensor_to_vae_latent, save_relative_pose, _filter2d, _resize_with_antialiasing, stratified_uniform, rand_cosine_interpolated, rand_log_normal
#for unet loading
from mv_diffusion import mvdream_diffusion_model
from mv_diffusion_SR import mvdream_diffusion_model as mvdream_diffusion_model_SR
from transformers import CLIPTextModel, CLIPTokenizer

#for dataset loading
from src.dataset import DatasetCfg, get_dataset
from src.dataset.data_module import worker_init_fn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from tqdm import tqdm
import imageio
import ipdb
from distutils.dir_util import copy_tree
from typing import Tuple, List
import torch.nn.functional as F

# from diffusion.utils.cam_utils import get_rays
# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from config.config import load_typed_root_config
    from config.global_cfg import set_cfg
    from src.misc.step_tracker import StepTracker

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def PIL2tensor(height,width,num_frames,masks,warps,logicalNot=False):
    channels = 3
    pixel_values = torch.empty((num_frames, channels, height, width))
    condition_pixel_values = torch.empty((num_frames, channels, height, width))
    masks_pixel_values = torch.ones((num_frames, 1, height, width))

    # input_ids
    prompt = ''

    for i, img in enumerate(masks):
        img = masks[i]
        img = img.convert('L') # make sure channel 1
        img_resized = img.resize((width, height)) # hard code here
        img_tensor = torch.from_numpy(np.array(img_resized)).float()

        # Normalize the image by scaling pixel values to [0, 1]
        img_normalized = img_tensor / 255
        mask_condition = (img_normalized > 0.9).float()
        #print(mask_condition, img_normalized)
        #assert False

        masks_pixel_values[i] = mask_condition

    for i, img in enumerate(warps):
        # Resize the image and convert it to a tensor
        img_resized = img.resize((width, height)) # hard code here
        img_tensor = torch.from_numpy(np.array(img_resized)).float()

        # Normalize the image by scaling pixel values to [-1, 1]
        img_normalized = img_tensor / 127.5 - 1

        img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

        if(logicalNot):
            img_normalized = torch.logical_not(masks_pixel_values[i])*(-1) + masks_pixel_values[i]*img_normalized
        condition_pixel_values[i] = img_normalized

    return [prompt], {
            'conditioning_pixel_values': condition_pixel_values, # [-1,1]
            'masks': masks_pixel_values# [0,1]
            }

def init_mvd(args):
    single_view = args.single_view
    super_resolution = args.super_resolution
    base_model_path = args.base_model_path

    if(single_view):
        mv_unet_path = base_model_path + "/unet/single/ema-checkpoint"
        print(mv_unet_path)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    else:
        mv_unet_path = base_model_path + "/unet/sparse/ema-checkpoint"
        print(mv_unet_path)
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")

    rgb_model = mvdream_diffusion_model(base_model_path,mv_unet_path,tokenizer,seed=12345)
    # mv_net_path = base_model_path + "/unet/SR/ema-checkpoint"
    # rgb_model_SR = mvdream_diffusion_model_SR(base_model_path,mv_unet_path,tokenizer,quantization=False,seed=12345)
    return rgb_model, None 

def get_image_files(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']

    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    image_names = [os.path.basename(file) for file in image_files]
    image_names = sorted(image_names, key=lambda x: int(x.split('.')[0].split('_')[-1]))

    return image_names

def read_train_imgs(path, height=512, width=512):
    image_names_ref = get_image_files(path)
    #print(path, image_names_ref)

    fimage = Image.open(os.path.join(path, image_names_ref[0]))
    #(width, height)= fimage.size
    #print(np.array(fimage).shape)
    _, _, channels  = np.array(fimage).shape
    result = []
    for imn in image_names_ref:
        #result.append(Image.open(os.path.join(source_imgs_dir + imn)))
        result.append(Image.open(os.path.join(path, imn)))
    num_frames = len(image_names_ref)

    condition_pixel_values = torch.empty((num_frames, channels, height, width))
    for i, img in enumerate(result):
        # Resize the image and convert it to a tensor
        img_resized = img.resize((width, height)) # hard code here
        img_tensor = torch.from_numpy(np.array(img_resized)).float()

        # Normalize the image by scaling pixel values to [-1, 1]
        img_normalized = img_tensor / 127.5 - 1

        img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

        condition_pixel_values[i] = img_normalized
    return condition_pixel_values.unsqueeze(0)

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)

def main(cfg_dict: DictConfig):
    # ipdb.set_trace()
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    args = cfg.diffusion
    cfg.dataset.image_shape = [args.height, args.width]
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,#args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        #log_with=args.report_to,
        project_config=accelerator_project_config,
        dispatch_batches=True,
        # kwargs_handlers=[ddp_kwargs]
    )
    #print(args.gradient_accumulation_steps, args.mixed_precision, args.report_to, accelerator_project_config)
    #assert False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

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
        generator = torch.Generator(
            device=accelerator.device).manual_seed(args.seed)
    else:
        generator = torch.Generator(device=accelerator.device)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    ## Load scheduler, tokenizer and models.
    #noise_scheduler = EulerDiscreteScheduler.from_pretrained(
    #    args.pretrained_model_name_or_path, subfolder="scheduler")
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

    #unet = STUnet.from_config(
    #    args.unet_config,
    #    low_cpu_mem_usage=True,
    #    variant="fp16",
    #)
        
    #if args.pretrain_unet is not None:
    #    unet.from_pretrained_skip_keys(
    #    args.pretrain_unet,
    #    # subfolder="unet",
    #    # low_cpu_mem_usage=True,
    #    # variant="fp16",
    #)
    #else:
    #    unet_svd = SVDUnet.from_pretrained(
    #        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
    #        subfolder="unet",
    #        low_cpu_mem_usage=True,
    #        variant="fp16",
    #    )

    #    unet.from_pretrained_model(unet_svd)
    rgb_model, rgb_model_SR = init_mvd(args)
    if not args.train_super_resolution:
        unet = rgb_model.unet.to(accelerator.device)
        vae = rgb_model.pipe.vae.to(accelerator.device)
        rgb_model.pipe.text_encoder = rgb_model.pipe.text_encoder.to(accelerator.device)

        tensor_to_vae_latent = vae.encode
        #latent_to_tensor = rgb_model.pipe.decode_latents
        encode_prompt = rgb_model.pipe._encode_prompt
        encode_image = rgb_model.pipe.encode_image

        get_wt = rgb_model.custom_decay_function_weight
        scaling = rgb_model.pipe.vae.config.scaling_factor
    else:
        unet = rgb_model_SR.unet.to(accelerator.device)
        vae = rgb_model_SR.pipe.vae.to(accelerator.device)
        tensor_to_vae_latent = vae.encode

        #latent_to_tensor = rgb_model_SR.pipe.decode_latents
        encode_prompt = rgb_model_SR.pipe._encode_prompt
        encode_image = rgb_model_SR.pipe.encode_image
        get_wt = rgb_model_SR.custom_decay_function_weight
        scaling = rgb_model_SR.pipe.vae.config.scaling_factor 


    
    #enable_ray_condition = unet.config.get("ray_condition", False)

    #use_image_embedding = unet.config.get("cross_attention_dim", None) is not None
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

        
    #accelerator = Accelerator()

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    # Move image_encoder and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    height_mvd = 512
    width_mvd = 512

    #if args.gradient_checkpointing:
    #    unet.enable_gradient_checkpointing()

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
    parameters_list = []

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
    ## build train dataset
    #if args.datasets_cfg.type == "single":
    #    train_dataset = get_dataset(cfg.dataset, "train", step_tracker)
    #    sampler = RandomSampler(train_dataset) if not isinstance(train_dataset, IterableDataset) else None
    #    train_dataloader = DataLoader(
    #        train_dataset,
    #        sampler=sampler,
    #        batch_size=args.per_gpu_batch_size, #cfg.data_loader.train.batch_size,
    #        shuffle=False, # randomsampler or IterableDataset do not need shuffle
    #        num_workers=args.num_workers,
    #        # worker_init_fn=worker_init_fn
    #    )
    #elif args.datasets_cfg.type == "concat":
    #    # ipdb.set_trace()
    #    from src.dataset.sampler import MultiNomialRandomSampler
    #    from src.dataset.sampler import ConcatDatasetWithIndex
    #    
    #    train_datasets = [get_dataset(dcfg, "train", step_tracker) for dcfg in args.datasets_cfg.datasets]
    #    if args.datasets_cfg.prob is not None:
    #        assert len(train_datasets) == len(args.datasets_cfg.prob), "The number of datasets and the number of probabilities should be the same."
    #        
    #    sampler = MultiNomialRandomSampler(train_datasets, p=args.datasets_cfg.prob, main_process=accelerator.is_main_process)
    #    train_dataset = ConcatDatasetWithIndex(train_datasets)
    #    train_dataloader = DataLoader(
    #        train_dataset,
    #        sampler=sampler,
    #        batch_size=args.per_gpu_batch_size, #cfg.data_loader.train.batch_size,
    #        shuffle=False, # randomsampler or IterableDataset do not need shuffle
    #        num_workers=args.num_workers,
    #        # worker_init_fn=worker_init_fn
    #    )
    #
    #else:
    #    raise ValueError(f"Unknown dataset type: {args.datasets_cfg.type}")
    
    print(f"Using seed {torch.seed()} for training.")
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    #num_update_steps_per_epoch = math.ceil(
    #    len(train_dataloader) / args.gradient_accumulation_steps)
    num_update_steps_per_epoch = 1000
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    ## Prepare everything with our `accelerator`.
    #unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
    #    unet, optimizer, lr_scheduler, train_dataloader
    #)

    pixel_values = read_train_imgs('/data/dylu/project/see4d/visualization/inputs/images').to(device = accelerator.device, dtype = weight_dtype)#[:,:4]
    pixel_values0 = pixel_values.to(accelerator.device)
    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, = accelerator.prepare(
        unet, optimizer, lr_scheduler,
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    #num_update_steps_per_epoch = math.ceil(
    #    len(train_dataloader) / args.gradient_accumulation_steps)
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
                "mode": "disabled",#cfg_dict.wandb.mode,
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
    #logger.info(f"  Num examples = {len(train_dataset)}")
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
        # accelerator.load_state(os.path.join(args.output_dir, path))
        # unet = STUnet.from_pretrained(os.path.join(args.output_dir, path))
        global_step = int(ckpt_path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * args.gradient_accumulation_steps)


    def batch_to_device(batch):
        for mode in ["context", "target"]:
            for k,v in batch[mode].items():
                batch[mode][k] = v.to(torch.float32).to(accelerator.device)
        if "static" in batch:
            for k,v in batch["static"].items():
                batch["static"][k] = v.to(torch.float32).to(accelerator.device)
                
        return batch

    #def get_wt(t, t_decay_end = 300, t_peak = 1000, v_decay_end = 0.8,b = 0.075):
    #    if t < t_decay_end:
    #        w_t = v_decay_end * (-b^(t_decay_end - t).exp())
    #    else:
    #        w_t = 1 - (1 - v_decay_end) * (t_peak - t) / (t_peak - t_decay_end)
    #    return w_t

    #def mask_pixels(imgs, downb = 0.1, upb = 0.5):
    #    threshold = np.random.uniform(downb, upb)
    #    #print(imgs.max(), imgs.min())
    #    #assert False
    #    prob = torch.rand_like(imgs[:,:1])#.mean(dim=1, keepdims=True)
    #    M = torch.where(prob>threshold*torch.ones_like(prob), torch.ones_like(prob), torch.zeros_like(prob))
    #    #print(M.shape, torch.ones_like(M))
    #    #print(M.mean(-1).mean(-1)/torch.ones_like(M).mean(-1).mean(-1))
    #    #assert False
    #    mask_imgs = M * imgs 
    #    return mask_imgs, M

    def mask_pixels(batch_images,  min_width: int = 1, max_width: int = 100):
        N, C, H, W = batch_images.shape

        # Create an empty mask batch
        masks_np = np.zeros((N, H, W), dtype=np.float32)  # Initialize all ones (white)

        # Generate random vertices (N, 2, 2), where each image gets two (x, y) coordinates
        vertices = np.random.randint(0, [W, H], size=(N, 2, 2))

        # Generate random line widths for each image
        widths = np.random.randint(min_width, max_width + 1, size=(N,))

        for i in range(N):
            # Create a separate mask image
            mask_pil = Image.new("L", (W, H), 255)  # Start as white (1.0)
            mask_draw = ImageDraw.Draw(mask_pil)

            # Draw black lines (0) on the mask
            mask_draw.line([tuple(vertices[i, 0]), tuple(vertices[i, 1])], fill=1, width=widths[i])

            # Convert mask back to NumPy (normalized to [0,1])
            masks_np[i] = np.array(mask_pil, dtype=np.float32) / 255.0

        # Convert masks to PyTorch format with shape (N, 1, H, W)
        masks = torch.from_numpy(masks_np).unsqueeze(1).to(batch_images.device, dtype = batch_images.dtype)

        # Apply mask to images (multiplication)
        processed_images = (1.0 + batch_images) * masks / 2.0  # Keeps image values in [-1,1] range

        return processed_images * 2.0 - 1.0, masks

    #def get_image_files(folder_path):
    #    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']

    #    image_files = []
    #    for ext in image_extensions:
    #        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    #    image_names = [os.path.basename(file) for file in image_files]

    #    return image_names

    def read_imgs(path, frame_id):
        source_imgs_dir = os.path.join(path, f"frame_{frame_id}", 'reference_images')#/dataset/htx/see4d/warps/outputs/cat_reverse_k3/frame_$i
        warp_root_dir = os.path.join(path, f"frame_{frame_id}", 'warp_images')
        os.makedirs(output_root_dir, exist_ok=True)
         
        height_mvd = 512
        width_mvd = 512
        masks_infer = []
        warps_infer = []
        input_names = []
        
        gt_num_b = 0
        mask2 = np.ones((height_mvd,width_mvd), dtype=np.float32)
        
        image_names_ref = get_image_files(source_imgs_dir)

        fimage = Image.open(os.path.join(source_imgs_dir, image_names_ref[0]))
        (width, height)= fimage.size
        for imn in image_names_ref:
            masks_infer.append(Image.fromarray(np.repeat(np.expand_dims(np.round(mask2*255.).astype(np.uint8),axis=2),3,axis=2)).resize((width_mvd, height_mvd)))
            warps_infer.append(Image.open(os.path.join(source_imgs_dir, imn)))
            input_names.append(imn)
            gt_num_b = gt_num_b + 1
         
        image_files = glob.glob(os.path.join(warp_root_dir, "warp_*"))
        image_names = [os.path.basename(image) for image in image_files]
        
        image_names.sort()
        print(image_names)
        for ins in image_names:
            warps_infer.append(Image.open(os.path.join(warp_root_dir, ins)))
            masks_infer.append(Image.open(os.path.join(warp_root_dir, ins.replace('warp','mask'))))
            input_names.append(ins)
        return masks_infer, warps_infer, input_names, gt_num_b, height_mvd, width_mvd, height, width

    def rand_ids(masks_infer, warps_infer, input_names, gt_num_b, ids = None, fnum = 4):
        if ids is None:
            nums = len(warps_infer)
            #ids = np.random.choice(list(range(nums-gt_num_b)), fnum)
            ids = list(range(nums-gt_num_b))[:fnum]
            ids = np.array([int(idi + gt_num_b) for idi in ids], dtype=int)
            print(ids)

        masks = masks_infer[:gt_num_b] + masks_infer[gt_num_b:gt_num_b+fnum-1]
        warps = warps_infer[:gt_num_b] + warps_infer[gt_num_b:gt_num_b+fnum-1]
        names = input_names[:gt_num_b] + input_names[gt_num_b:gt_num_b+fnum-1]
        
        return masks, warps, names, ids

    def save_mask_tensor_as_images(mask_tensor: torch.Tensor, output_dir: str, prefix: str = 'mask'):
        """
        Save a mask tensor of shape [B, 1, H, W] as grayscale images.

        Args:
        - mask_tensor: torch.Tensor, shape [B, 1, H, W], values should be in range [0, 1] or boolean
        - output_dir: directory to save the images
        - prefix: filename prefix for saved images (default: 'mask')
        """
        os.makedirs(output_dir, exist_ok=True)

        # Convert to float if the mask is in boolean type
        if mask_tensor.dtype == torch.bool:
            mask_tensor = mask_tensor.float()

        for i in range(mask_tensor.size(0)):
            save_path = os.path.join(output_dir, f'{prefix}_{i:02d}.png')
            save_image(mask_tensor[i], save_path)

    #def read_train_imgs(path, height=512, width=512):
    #    image_names_ref = get_image_files(path)
    #    #print(path, image_names_ref)

    #    fimage = Image.open(os.path.join(path, image_names_ref[0]))
    #    #(width, height)= fimage.size
    #    #print(np.array(fimage).shape)
    #    _, _, channels  = np.array(fimage).shape
    #    result = []
    #    for imn in image_names_ref:
    #        #result.append(Image.open(os.path.join(source_imgs_dir + imn)))
    #        result.append(Image.open(os.path.join(path, imn)))
    #    num_frames = len(image_names_ref)

    #    condition_pixel_values = torch.empty((num_frames, channels, height, width))
    #    for i, img in enumerate(result):
    #        # Resize the image and convert it to a tensor
    #        img_resized = img.resize((width, height)) # hard code here
    #        img_tensor = torch.from_numpy(np.array(img_resized)).float()

    #        # Normalize the image by scaling pixel values to [-1, 1]
    #        img_normalized = img_tensor / 127.5 - 1

    #        img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

    #        condition_pixel_values[i] = img_normalized
    #    return condition_pixel_values.unsqueeze(0)
        
            
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    # ipdb.set_trace()
    inference_times = 0
    #total_values = torch.tensor([0]*len(args.datasets_cfg.prob))
    #average_motion = torch.zeros(len(args.datasets_cfg.prob)).to(accelerator.device)
    step_tracker.set_step(global_step)
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        #pixel_values = read_train_imgs('/dataset/htx/see4d/inputs/img/').to(device = unet.device, dtype = torch.float16)[:,:4]
        bsz = pixel_values.shape[0]
        width, height = pixel_values.shape[-1], pixel_values.shape[-2]
        num_frames = 16
        #print(pixel_values.shape)
        #print(num_frames)
        #assert False
        #for step, batch in enumerate(train_dataloader):
        idlist = list(np.arange(0, pixel_values.shape[1]))
        for step in range(1000):
            idi = np.array([11] + list(np.random.choice(idlist, num_frames - 1, replace=False)))
            #pixel_values0 = pixel_values[:,idi]
            # # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            step_tracker.set_step(global_step)
            #total_values[batch['dataset_idx'][0]] += 1
            with accelerator.accumulate(unet):
                ### map to device and fp32 for splat image
                #batch = batch_to_device(batch)
                ## ipdb.set_trace()
                #matches = batch['context_target_correspondence'].long() # B, match
                ## ipdb.set_trace()
                #indicator = batch['indicator'].long() # B, F
                ## NOTE indicator is always 4d = 2
                #indicator = torch.ones_like(indicator) * 2 # B, F

                ## Add sanity check
                #if (global_step == 1 or global_step % 1000 == 0) and accelerator.is_main_process:
                #    save_path = os.path.join(args.output_dir, f"sanity-check/step-{global_step}.mp4")
                #    os.makedirs(os.path.join(args.output_dir, f"sanity-check"), exist_ok=True)
                #    save_image_video(batch['context']['image'][0].data.cpu(), batch['target']['image'][0].data.cpu() , save_path)
                    
                #if (global_step % 1000 == 0):
                #    tqdm.write(f"Step {global_step} | Dataset sample values {total_values / (global_step+1e-3)}")
                #    tqdm.write(f"Step {global_step} Process {accelerator.process_index} Scene : {batch['scene'][0]}")

                # first, convert images to latent space.
                #pixel_values = batch["target"]["image"].to(weight_dtype)
                pixel_values = pixel_values0[:,idi]
                video_length = pixel_values.size(1)
                # normalize to -1 to 1
                #pixel_values = pixel_values * 2 - 1
                #print(pixel_values.shape, num_frames)
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w", f=num_frames)

                latents = vae.encode(pixel_values).latent_dist.sample() * scaling # b f c h w
                actual_num_frames = latents.shape[1]
                #warp_latents = tensor_to_vae_latent(mask_latents(pixel_values)) # b f c h w
                warp_pixels, masks = mask_pixels(pixel_values) 
                #save_mask_tensor_as_images((warp_pixels+1)/2, output_dir = '/dataset/htx/see4d/outputs/purse_mask')
                #print(masks.shape)
                #assert False

                warp_latents = vae.encode(warp_pixels).latent_dist.sample() * scaling

                #masks = rearrange(masks, "b f c h w -> (b f) c h w", f=latents.shape[1])# [b*f, c, h, w]
                mask0 = ((rearrange(torch.ones_like(pixel_values), "(b f) c h w -> b f c h w", f=num_frames)+1.0)/2 > 0.9).float().mean(2, keepdim=True)
                mask0 = rearrange(mask0, "b f c h w -> (b f) c h w", f=num_frames)
                mask_latents = torch.nn.functional.interpolate(
                    masks,
                    size=(
                        height // 8,
                        width // 8
                    )
                ).to(weight_dtype).to(accelerator.device)# [b*f, c, h, w]

                mask0 = torch.nn.functional.interpolate(
                    mask0,
                    size=(
                        height // 8,
                        width // 8
                    )
                ).to(weight_dtype).to(accelerator.device)# [b*f, c, h, w]

                #warp_latents = mask_latents * warp_latents

                warp_latents = rearrange(warp_latents, "(b f) c h w -> b f c h w", f=num_frames)
                latents = rearrange(latents, "(b f) c h w -> b f c h w", f=num_frames)
                mask_latents = rearrange(mask_latents, "(b f) c h w -> b f c h w", f=num_frames)

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
                timesteps = timesteps.long()

                w_t = get_wt(timesteps.float())
                w_t = w_t.view(w_t.shape[0], 1, 1, 1).to(w_t.dtype)

                noise = torch.randn_like(latents)
                warp_noisy_latents = noise_scheduler.add_noise(warp_latents, noise, timesteps)
                inp_noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                warp_noisy_latents = w_t * warp_noisy_latents + (1 - w_t) * inp_noisy_latents

                batch_size = latents.size(0)
                # ipdb.set_trace()
                #motion = torch.ones(batch_size).to(latents.device, latents.dtype) # b

                # Get the target for loss depending on the prediction type
                #if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                #    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                #first_sample_dataset_idx = batch['dataset_idx'][0]
                #average_motion[first_sample_dataset_idx] = average_motion[first_sample_dataset_idx]*total_values[first_sample_dataset_idx]/(total_values[first_sample_dataset_idx]+1) + motion[0]/(total_values[first_sample_dataset_idx]+1)
    
                ## if use camera
                # video not use camera
                #camera_indicator = torch.ones(batch_size).to(latents.device, latents.dtype)
                    
                ## Get the text embedding for conditioning.
                ### NOTE here is clip image embedding
                #context_pixel_values = batch["context"]["image"].to(weight_dtype) 
                #num_context_views = context_pixel_values.size(1)

                ##encoder_hidden_states = torch.zeros((batch_size*num_context_views, 128)).to(latents.device, latents.dtype) # bv, 1024
                ##encoder_hidden_states = rearrange(encoder_hidden_states, "(b v) c -> b v c", v=num_context_views)
                #

                ### NOTE projection_class_embeddings_input_dim is the input feature dim of time embedding linear layer
                ### it should be addition_time_embed_dim * num_additional_values
                #added_time_ids = torch.stack([torch.zeros_like(motion), camera_indicator], dim=-1).to(latents.device, encoder_hidden_states.dtype) # b,2    
                #added_time_ids = added_time_ids.unsqueeze(1).repeat_interleave(video_length, dim=1) # b,f,2

                ## Sample noise that we'll add to the latents
                ##noise = torch.randn_like(latents)
                #
                ## conditional_pixel_values = pixelsplat_color # bs,f,3,h,w # -1 to 1
                #
                #cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
                #noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1
                #cond_sigmas = cond_sigmas[:, None, None, None, None]
                #conditional_latents = torch.zeros_like(latents) # b,f,c,h,w
                #
                ### Sample a random timestep for each image
                ### P_mean=0.7 P_std=1.6
                #sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                ### Add noise to the latents according to the noise magnitude at each timestep
                ### (this is the forward diffusion process)
                #sigmas = sigmas[:, None, None, None, None] # B,1,1,1,1
                #noisy_latents = latents + noise * sigmas

                #timesteps = torch.Tensor(
                #    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                #inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)
                context_mask = torch.zeros(bsz, video_length).to(accelerator.device) # b,f, 0 denote predicted views and 1 denote condition views
                #context_mask.scatter_(1, matches.to(context_mask.device), 1)
                if args.gt_num:
                    context_mask[:, :args.gt_num] = 1

                ### expand context mask to the same size of inp_noisy_latents
                context_mask_map = repeat(context_mask, "b f -> b f c h w", c=1, h=inp_noisy_latents.size(-2), w=inp_noisy_latents.size(-1)) # b,f,1,h,w
                context_latents = latents

                if args.conditioning_dropout_prob is not None and random.random() < args.conditioning_dropout_prob:
                    # ipdb.set_trace()
                    #context_latents = torch.zeros_like(latents)
                    ## cfg zero means using zero as context
                    ## if not, use noise as context
                    #if not args.cfg_zero:
                    #    context_mask_map = torch.zeros_like(context_mask_map)
                    warp_noisy_latents = torch.zeros_like(warp_noisy_latents)
                    mask_latents_zero = torch.zeros_like(mask_latents)
                    mask_latents = mask_latents_zero * (1 - context_mask_map) + mask_latents * context_mask_map

                inp_noisy_latents = inp_noisy_latents * (1 - context_mask_map) + context_latents * context_mask_map
                warp_noisy_latents = warp_noisy_latents * (1 - context_mask_map) + context_latents * context_mask_map
                mask_latents = mask_latents * (1 - context_mask_map) + mask0 * context_mask_map

                #if args.conditioning_dropout_prob is not None:
                #    # ipdb.set_trace()
                #    random_p = torch.rand(
                #        bsz, device=latents.device, generator=generator)

                #    # image_mask_dtype = conditional_latents.dtype
                #    image_mask = 1 - (
                #        (random_p < args.conditioning_dropout_prob).to(conditional_latents.dtype)
                #    )
                #        
                #    image_mask = image_mask.reshape(bsz, 1, 1, 1, 1)
                #    # Final image conditioning.
                #    conditional_latents = image_mask * conditional_latents
                    
                # Concatenate the `conditional_latents` with the `noisy_latents`.
                # NOTE in my case, no need to repeat
                # conditional_latents = conditional_latents.unsqueeze(
                #     1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                #print(inp_noisy_latents.shape,  warp_noisy_latents.shape, mask_latents.shape)
                latent_model_input = torch.cat(
                    [inp_noisy_latents, warp_noisy_latents, mask_latents], dim=2) # b,f,c+c_img+c_ray+1,h,w


                ## check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                #target = latents
                #model_pred = unet(
                #    inp_noisy_latents, 
                #    timesteps, 
                #    encoder_hidden_states, 
                #    added_time_ids=added_time_ids, 
                #    indicator=indicator,
                #    motion_ids=motion_ids,
                #    time_steps=batch['time_steps'].to(latents.device, latents.dtype),
                #    ).sample
                with torch.no_grad():
                    num_images_per_prompt = 1
                    prompt_embeds = encode_prompt('', accelerator.device, num_images_per_prompt, False, None)
                    prompt_embeds = prompt_embeds.repeat([bsz, 1, 1])

                    image_embeds_neg, image_embeds_pos = encode_image(pixel_values, accelerator.device, num_images_per_prompt)
                    #image_embeds_neg, image_embeds_pos = encode_image(rearrange(pixel_values, "b f c h w -> (b f) c h w", f=latents.shape[1])\
                    #        , latents.device, num_images_per_prompt)
                    image_embeds_pos = rearrange(image_embeds_pos, "(b f) l w -> b f l w", f=num_frames)[:,0:1]
                    image_embeds_pos = image_embeds_pos.squeeze(1)#.repeat([1, num_frames, 1, 1])

                latent_model_input = latent_model_input.reshape([bsz*num_frames, 9, latents.shape[-2], latents.shape[-1]])
                #assert False
                #print(latent_model_input.shape, (prompt_embeds + image_embeds_pos).unsqueeze(1).repeat(1, num_frames, 1, 1).shape)
                useless_embeds = (prompt_embeds + image_embeds_pos).unsqueeze(1).repeat(1, num_frames, 1, 1)
                useless_embeds = rearrange(useless_embeds, "b f l w -> (b f) l w", f=num_frames)#.detach()
                useless_embeds = useless_embeds.to(dtype=weight_dtype)

                #print(latent_model_input.shape, timesteps.shape, useless_embeds.shape, num_frames)
                #assert False

                unet_inputs = {
                    'x': latent_model_input.detach(),# torch.Size([num_frames, 5, 32, 32])
                    'timesteps': torch.tensor(timesteps, dtype=latent_model_input.dtype),# torch.Size([num_frames])
                    'context': useless_embeds.detach(),#.to(unet.device),
                    'num_frames': num_frames,# 4
                    'camera': None,#
                }
                model_pred = unet.forward(**unet_inputs)# [6, 4, 32, 32]
                model_pred = model_pred.reshape([bsz, num_frames, -1, latents.shape[-2], latents.shape[-1]])#[:,:,:4]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float()[:,:,:4], target.float(), reduction="mean") #+ 0.0 * model_pred.float()[:,:,4:].mean()
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float()[:,:,:4], target.float().to(model_pred.device), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                loss += 0.0 * sum(torch.norm(param, p=2) ** 2 for param in unet.parameters())

                ## Denoise the latents
                #c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                #c_skip = 1 / (sigmas**2 + 1)
                #denoised_latents = model_pred * c_out + c_skip * noisy_latents
                #weighting = (1 + sigmas ** 2) * (sigmas**-2.0) # B,1,1,1,1
                ### weighting is set to zeros for matched latents
                #weighting = weighting.repeat_interleave(video_length, dim=1) # B,F,1,1,1
                ## weighting[batch_indices, matches] = 0
                #weighting = weighting * (1 - context_mask[:, :, None, None, None]) # context_mask, b f -> b f 1 1 1, context_mask = 1 denotes condition view, and we do not calculate loss for condition view

                ## MSE loss
                #loss = torch.mean(
                #    (weighting.float() * (denoised_latents.float() -
                #     target.float()) ** 2).reshape(target.shape[0], -1),
                #    dim=1,
                #)
                #loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                #loss.backward()
                if accelerator.sync_gradients:
                    #accelerator.unscale_gradients()
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    #"obj4D-data": total_values[3:4].sum() if len(total_values) > 5 else 0,
                    #"webvid4d-data": total_values[:3].sum() if len(total_values) > 5 else 0,
                    #"3D-data": total_values[4:-1].sum() if len(total_values) > 5 else 0,
                    #"video-data": total_values[-1:].sum() if len(total_values) > 5 else 0,
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
                            copy_tree(save_path, save_path.replace("checkpoint", "ckpt"))
                            
                    # sample images!
                    # if (
                    #     (global_step % args.validation_steps == 0)
                    #     or (global_step == 1)
                    # ):
                    
                    if True:
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        inference_times += 1
                        # create pipeline
                        
                        #pipeline = create_pipeline('genxd_time')(
                        #    unet=accelerator.unwrap_model(unet),
                        #    vae=accelerator.unwrap_model(vae),
                        #    scheduler=noise_scheduler,
                        #)
                        #

                        #pipeline = pipeline.to(device=accelerator.device) ## NOTE here we canno to dtype, or there will be error in next optimization step
                        #pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ): 
                        #    # import copy
                        #    ## Evaluate each dataset once
                        #    if args.datasets_cfg.type == "single":
                        #        args.datasets_cfg.prob = [1]
                        #        args.datasets_cfg.datasets = [cfg.dataset]
                        ##        
                        #    for dataset_index in range(len(args.datasets_cfg.prob)):
                        #        print(f"Running validation for dataset {dataset_index} / {len(args.datasets_cfg.prob)}")
                        #        evaluation_dataset = get_dataset(args.datasets_cfg.datasets[dataset_index], "val", step_tracker)
                        #        eval_sampler = RandomSampler(evaluation_dataset)
                        #        evaluation_dataloader = DataLoader(
                        #            evaluation_dataset,
                        #            sampler=eval_sampler,
                        #            batch_size=1, #cfg.data_loader.train.batch_size,
                        #            shuffle=False,
                        #            num_workers=0,
                        #            worker_init_fn=worker_init_fn
                        #        )
                        #        eval_batch = next(iter(evaluation_dataloader))

                        #        num_frames = args.num_frames
                        #        video_output = pipeline(
                        #            eval_batch,
                        #            # accelerator.unwrap_model(pixelsplat_model),
                        #            height=args.height,
                        #            width=args.width,
                        #            num_frames=num_frames,
                        #            decode_chunk_size=8,
                        #            motion_bucket_id=127,
                        #            fps=7,
                        #            noise_aug_strength=0.02,
                        #            # generator=generator,
                        #            output_type="pt",
                        #            splat_dropout=False,
                        #            use_image_embedding=False,
                        #            indicator=eval_batch['indicator'][0],
                        #            motion_strength=1.0,
                        #            use_motion_embedding=args.use_motion_embedding,
                        #            single_view_inference=False,
                        #            temporal_time_steps=eval_batch['time_steps'][0],
                        #        )
                               ids = None
                               fnum = 1

                               for i in range(fnum):
                                   rgb_model.unet = unet
                                   rgb_model.vae = vae
                                   output_root_dir = os.path.join('/data/dylu/project/see4d/visualization/outputs/purse','val300', f'frame_{i}')
    
                                   masks_infer, warps_infer, input_names, gt_num_b, height_mvd, width_mvd, vheight, vwidth = read_imgs(args.val_dir, i)
                                   masks_infer_batch, warp_infer_batch, input_names_batch, ids = rand_ids(masks_infer, warps_infer, input_names, gt_num_b, ids = ids, fnum = 16)
                                   #save_mask_tensor_as_images(batch['masks'][0], output_dir = '/dataset/htx/see4d/outputs/purse_mask')
                                   
                                   prompt, batch = PIL2tensor(height_mvd,width_mvd,len(masks_infer_batch),masks_infer_batch,warp_infer_batch,logicalNot=False)
                                   images_predict_batch = rgb_model.inference_next_frame(prompt,batch,len(masks_infer_batch),height_mvd,width_mvd,gt_num_frames=gt_num_b,output_type='pil')
                                   #save_mask_tensor_as_images(batch['masks'][0], output_dir = '/dataset/htx/see4d/outputs/purse_mask')
                                   #assert False

                                   images_predict = []
                                   images_mask_p = []
                                   images_predict_names = []


                                   for jj in range(gt_num_b,len(images_predict_batch)):
                                       images_predict.append(images_predict_batch[jj])
                                       #print(batch['masks'].shape)
                                       images_mask_p.append(batch['masks'][0][jj][0].cpu().numpy())
                                       images_predict_names.append(input_names_batch[jj])
                                   print(images_predict_names)
                           
                                   for jj in range(len(images_predict)):
                                       images_predict[jj].resize((vwidth, vheight)).save(os.path.join(output_root_dir,"predict_{}.jpg".format(images_predict_names[jj]))) 
                                       #images_predict[jj].save(os.path.join(output_root_dir,"predict_{}.jpg".format(images_predict_names[jj])))
                                
                                #video_frames = video_output.frames[0] # f,c,h,w
                                #context_image = eval_batch['context']['image'][0].data.cpu() # 0-1
                                #target_video = eval_batch['target']['image'][0] # 0-1 f,c,h,w
                                #if video_output.renderings is not None:
                                #    renderings = video_output.renderings[0]
                                #    save_video = torch.cat([target_video, renderings, video_frames], dim=-1) # f,c,h,3w
                                #else:
                                #    save_video = torch.cat([target_video, video_frames], dim=-1) # f,c,h,3w
                                #out_file = os.path.join(
                                #    val_save_dir,
                                #    f"step_{global_step}_val_img_dataset_{dataset_index}.mp4",
                                #)
                                #vidnp = save_image_video(context_image, save_video.data.cpu(), out_file, fps=4)


                        #del pipeline
                        torch.cuda.empty_cache()
                        assert False

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
