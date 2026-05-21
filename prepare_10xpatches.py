import os
import json
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn.functional as F

# Using allow_compressed_pixels to handle large images if necessary
Image.MAX_IMAGE_PIXELS = None 

def bandpass_filter_staig_gpu(
    img_tensor: torch.Tensor,
    low: int = 245,
    high: int = 275,
    pre_blur: bool = True,
    post_blur: bool = True,
    pre_ksize: int = 5,
    post_ksize: int = 15
):
    device = img_tensor.device
    gray = img_tensor.mean(dim=0, keepdim=True)  # (1, H, W)

    if pre_blur:
        k = pre_ksize // 2
        kernel = torch.exp(
            -((torch.arange(-k, k + 1, device=device).float()) ** 2)[None, :] / (2 * (k / 2) ** 2)
        )
        kernel2d = (kernel.T @ kernel)
        kernel2d = kernel2d / kernel2d.sum()
        kernel2d = kernel2d.expand(1, 1, -1, -1)
        gray = F.conv2d(gray.unsqueeze(0), kernel2d, padding=k).squeeze(0)

    f = torch.fft.fft2(gray)
    fshift = torch.fft.fftshift(f)
    h, w = gray.shape[1:]
    cy, cx = h // 2, w // 2

    mask = torch.zeros((h, w), dtype=torch.complex64, device=device)
    mask[cy - high:cy + high, cx - high:cx + high] = 1.0
    mask[cy - low:cy + low, cx - low:cx + low] = 0.0

    filtered = fshift * mask
    ishift = torch.fft.ifftshift(filtered)
    img_back = torch.fft.ifft2(ishift).abs()

    if post_blur:
        k = post_ksize // 2
        kernel = torch.exp(
            -((torch.arange(-k, k + 1, device=device).float()) ** 2)[None, :] / (2 * (k / 2) ** 2)
        )
        kernel2d = (kernel.T @ kernel)
        kernel2d = kernel2d / kernel2d.sum()
        kernel2d = kernel2d.expand(1, 1, -1, -1)
        img_back = F.conv2d(img_back.unsqueeze(0), kernel2d, padding=k).squeeze(0)

    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8)
    img_back = img_back.repeat(3, 1, 1)

    return img_back


def extract_and_save_patches(
    spatial_dir,
    output_dir,
    patch_scale=3.5,
    resize1=512,
    resize2=224
):
    spatial_dir = Path(spatial_dir)
    output_dir = Path(output_dir)

    json_path = spatial_dir / "scalefactors_json.json"
    if not json_path.exists():
        print(f"Warning: No scalefactors found in {spatial_dir}")
        return

    with open(json_path, "r") as f:
        d = json.load(f)["fiducial_diameter_fullres"]
    
    patch_size = int(math.ceil(patch_scale * d))
    half = patch_size // 2

    img_path = spatial_dir / "tissue_full_image.tif"
    
    if img_path.exists():
        print(f"Loading full-res image: {img_path.name}")
        img = Image.open(img_path).convert("RGB")
    else:
        tif_files = list(spatial_dir.glob("*.tif"))
        if not tif_files:
            print(f"Error: No 'tissue_full_image.tif' or other .tif found in {spatial_dir}")
            return
        print(f"Loading fallback TIF: {tif_files[0].name}")
        img = Image.open(tif_files[0]).convert("RGB")
        
    W, H = img.size
    print(f"Image Size: {W}x{H}")

    coords_path = spatial_dir / "tissue_positions_list.csv"
    if not coords_path.exists():
        coords_path = spatial_dir / "tissue_positions.csv"
        coords = pd.read_csv(coords_path) 
        if 'barcode' not in coords.columns and 'in_tissue' not in coords.columns:
             coords = pd.read_csv(coords_path, header=None)
             coords.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
    else:
        coords = pd.read_csv(coords_path, header=None)
        coords.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']

    if 'in_tissue' in coords.columns:
        coords = coords[coords['in_tissue'] == 1]
    else:
        coords = coords[coords.iloc[:, 1] == 1]

    sample_id = spatial_dir.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((resize1, resize1)),
        T.ToTensor()
    ])

    saved, skipped, missing = 0, 0, 0

    for _, row in tqdm(coords.iterrows(), total=len(coords), desc=f"Processing {sample_id}"):
        barcode = row['barcode'] if 'barcode' in row else row.iloc[0]
        pt_path = output_dir / f"{barcode}.pt"

        if pt_path.exists():
            try:
                t = torch.load(pt_path)
                if (
                    not isinstance(t, torch.Tensor) or
                    torch.isnan(t).any() or
                    t.shape != (3, resize2, resize2) or
                    t.abs().sum() == 0 
                ):
                    # print(f"Invalid patch detected: {barcode}, regenerating...")
                    pt_path.unlink() 
                else:
                    skipped += 1
                    continue
            except Exception as e:
                print(f"Corrupt file {barcode}: {e}, regenerating...")
                pt_path.unlink()

        raw_x = row['pxl_col'] if 'pxl_col' in row else row.iloc[5]
        raw_y = row['pxl_row'] if 'pxl_row' in row else row.iloc[4]
        
        x, y = int(raw_x), int(raw_y)
        
        if (x - half < 0) or (x + half > W) or (y - half < 0) or (y + half > H):
            missing += 1
            continue

        try:
            patch = img.crop((x - half, y - half, x + half, y + half))
            
            tensor = transform(patch).to("cuda")

            filtered = bandpass_filter_staig_gpu(
                tensor, 
                low=245, 
                high=275, 
                pre_blur=True, 
                post_blur=True
            )

            final = T.Resize((resize2, resize2))(filtered).cpu()
            
            torch.save(final, pt_path)
            saved += 1

        except Exception as e:
            print(f"Failed to process {barcode}: {e}")
            missing += 1

    print(f"Sample: {sample_id} | Saved: {saved} | Skipped (Exists): {skipped} | Failed/Edge: {missing}")


if __name__ == "__main__":
    
    root_dir = Path("../RAW_SLICE/test10x") 
    
    spatial_dirs = list(root_dir.glob("*/spatial"))
    
    #spatial_dirs = [d for d in spatial_dirs if "Human_Breast_Cancer" in str(d)]
    #spatial_dirs = [d for d in spatial_dirs if "Mouse_Brain_Anterior" in str(d)]
    #spatial_dirs = [d for d in spatial_dirs if "Mouse_Brain_Coronal" in str(d)]
    #spatial_dirs = [d for d in spatial_dirs if "Mouse_Brain_Posterior" in str(d)]
    #spatial_dirs = [d for d in spatial_dirs if "WS_PLA_S9101767" in str(d)]
    #spatial_dirs = [d for d in spatial_dirs if "WS_PLA_S9101765" in str(d)]
    #spatial_dirs = [d for d in spatial_dirs if "WS_PLA_S9101764" in str(d)]
    

    print(f"Found {len(spatial_dirs)} samples: {[d.parent.name for d in spatial_dirs]}")
    
    output_dir_base = Path("patch_cache_gpu")

    for spatial_dir in spatial_dirs:
        sample_id = spatial_dir.parent.name
        output_dir = output_dir_base / sample_id
        extract_and_save_patches(spatial_dir, output_dir)