import os
import glob
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import inspect
import random

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

        img_normalized = img_tensor / 255.0

        img_normalized = img_normalized.permute(2, 0, 1)  # For RGB images

        condition_pixel_values[i] = img_normalized
    return condition_pixel_values.unsqueeze(0)

def mask_pixels(batch_images,  min_width: int = 20, max_width: int = 100):
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
        mask_draw.line([tuple(vertices[i, 0]), tuple(vertices[i, 1])], fill=0, width=widths[i])

        # Convert mask back to NumPy (normalized to [0,1])
        masks_np[i] = np.array(mask_pil, dtype=np.float32) / 255.0

    # Convert masks to PyTorch format with shape (N, 1, H, W)
    masks = torch.from_numpy(masks_np).unsqueeze(1).to(batch_images.device, dtype = batch_images.dtype)

    # Apply mask to images (multiplication)
    processed_images = batch_images * masks   # Keeps image values in [-1,1] range

    return processed_images, masks

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

def prepare_extra_step_kwargs(generator, eta, noise_scheduler):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(
        inspect.signature(noise_scheduler.step).parameters.keys()
    )
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(
        inspect.signature(noise_scheduler.step).parameters.keys()
    )
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


def random_edge_mask(batch_masks, max_frac=0.25):
    """
    Randomly zero-out bands along up to four image edges,
    using the **same** mask for every image in the batch.

    Parameters
    ----------
    batch_masks : torch.Tensor
        Shape (N, 1, H, W). Will be **modified in-place**.
    max_frac : float
        Maximum band width as a fraction of the spatial size (≤ 0.5).  
        Default 0.25 → up to 512/4 = 128 px when H=W=512.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    N, _, H, W = batch_masks.shape
    assert H == W, "Only square inputs assumed here"
    max_band = int(H * max_frac)

    # --- 1. Decide which edges to mask -------------------------------------
    edges = ["top", "bottom", "left", "right"]
    k      = random.randint(1, len(edges))          # how many edges to mask
    chosen = random.sample(edges, k)

    # --- 2. Build a single boolean mask ------------------------------------
    # start with all ones (keep)
    edge_mask = torch.ones(1, 1, H, W, dtype=torch.bool,
                           device=batch_masks.device)

    for edge in chosen:
        band = random.randint(0, max_band)          # width in pixels
        if band == 0:
            continue                                # nothing to do
        if edge == "top":
            edge_mask[:, :, :band, :] = False
        elif edge == "bottom":
            edge_mask[:, :, -band:, :] = False
        elif edge == "left":
            edge_mask[:, :, :, :band] = False
        elif edge == "right":
            edge_mask[:, :, :, -band:] = False

    # --- 3. Apply it to every image in the batch ---------------------------
    # broadcast  (1,1,H,W)  → (N,1,H,W)
    batch_masks.masked_fill_(~edge_mask, 0)

    return batch_masks      # optionally return the mask for inspection
