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

def bandpass_filter_torch(img_tensor, low=245, high=275):
    gray = img_tensor.mean(dim=0, keepdim=True)
    f = torch.fft.fft2(gray)
    fshift = torch.fft.fftshift(f)
    h, w = gray.shape[1:]
    cy, cx = h // 2, w // 2

    mask = torch.zeros((h, w), dtype=torch.complex64, device=img_tensor.device)
    mask[cy - high:cy + high, cx - high:cx + high] = 1.0
    mask[cy - low:cy + low, cx - low:cx + low] = 0.0

    filtered = fshift * mask
    ishift = torch.fft.ifftshift(filtered)
    img_back = torch.fft.ifft2(ishift).abs()
    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-8)
    img_back = img_back.repeat(3, 1, 1)
    img_back = F.avg_pool2d(img_back.unsqueeze(0), kernel_size=15, stride=1, padding=7).squeeze(0)
    return img_back


def extract_and_save_patches(spatial_dir, output_dir, patch_scale=3.5, resize1=512, resize2=224):
    spatial_dir = Path(spatial_dir)
    output_dir = Path(output_dir)
    with open(spatial_dir / "scalefactors_json.json", "r") as f:
        d = json.load(f)["fiducial_diameter_fullres"]
    patch_size = int(math.ceil(patch_scale * d))
    half = patch_size // 2
    tif_files = list(spatial_dir.glob("*.tif"))
    if not tif_files:
        print(f"No .tif found in {spatial_dir}")
        return
    img = Image.open(tif_files[0]).convert("RGB")
    W, H = img.size
    coords = pd.read_csv(spatial_dir / "tissue_positions_list.csv", header=None)
    coords.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
    coords = coords[coords['in_tissue'] == 1]
    sample_id = spatial_dir.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = T.Compose([
        T.Resize((resize1, resize1)),
        T.ToTensor()
    ])
    saved, skipped, missing = 0, 0, 0
    for _, row in tqdm(coords.iterrows(), total=len(coords), desc=output_dir.name):
        barcode = row['barcode']
        pt_path = output_dir / f"{barcode}.pt"
        if pt_path.exists():
            try:
                loaded = torch.load(pt_path)
                if (
                        not isinstance(loaded, torch.Tensor) or
                        torch.isnan(loaded).any() or
                        loaded.shape != (3, resize2, resize2) or
                        loaded.abs().sum() == 0
                ):
                    print(f"Invalid patch {pt_path.name}, regenerating...")
                    pt_path.unlink()
                else:
                    skipped += 1
                    continue
            except Exception as e:
                print(f"Error loading {pt_path.name}, regenerating... ({e})")
                pt_path.unlink()
        x, y = int(row['pxl_col']), int(row['pxl_row'])
        if (x - half < 0) or (x + half > W) or (y - half < 0) or (y + half > H):
            print(f"Skip spot {barcode} due to image edge (x={x}, y={y})")
            continue
        box = (x - half, y - half, x + half, y + half)
        try:
            patch = img.crop(box).resize((resize1, resize1))
            tensor = transform(patch).to("cuda")
            filtered = bandpass_filter_torch(tensor).cpu()
            final = T.Resize((resize2, resize2))(filtered)
            torch.save(final, pt_path)
            saved += 1
        except Exception as e:
            print(f"Failed {barcode}: {e}")
            missing += 1
            continue
    total = len(coords)
    print(f"{sample_id} | Saved: {saved}/{total} | Skipped (edge/existing): {skipped} | Failed: {missing}")

if __name__ == "__main__":
    root_dir = Path("../RAW_SLICE/DLPFC")
    spatial_dirs = list(root_dir.glob("*/spatial"))
    print(f"Found {len(spatial_dirs)} samples.")
    for spatial_dir in spatial_dirs:
        sample_id = spatial_dir.parent.name
        output_dir = Path("patch_cache_gpu") / sample_id
        extract_and_save_patches(spatial_dir, output_dir)

