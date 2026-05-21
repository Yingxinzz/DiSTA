import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # 引入 transforms
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from dinov2.models.vision_transformer import vit_giant2
from PIL import Image
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x) 
        return F.normalize(x, dim=-1)

class CachedPatchDataset(Dataset):
    def __init__(self, raw_root, cache_dir, transform=None, only_sample=None, verbose=True, return_dual_view=False):
        
        self.items = []
        self.transform = transform
        self.return_dual_view = return_dual_view
        
        raw_root = Path(raw_root).resolve()
        cache_dir = Path(cache_dir).resolve()
        
        spatial_dirs = list(raw_root.glob("*/spatial"))
        
        if only_sample:
            spatial_dirs = [d for d in spatial_dirs if d.parent.name == only_sample]
            
        if verbose:
            print(f"--- CachedPatchDataset Initialization ---")
            print(f"Found {len(spatial_dirs)} slices under {raw_root}")
            
        total_in_tissue_spots = 0
        
        for spatial_dir in spatial_dirs:
            sample_name = spatial_dir.parent.name
            
            coord_path = spatial_dir / "tissue_positions_list.csv"
            if not coord_path.exists():
                coord_path = spatial_dir / "tissue_positions.csv"
            
            if not coord_path.exists():
                if verbose: print(f"Warning: No coords found for {sample_name}")
                continue

            coords = None
            try:
                coords = pd.read_csv(coord_path, header=None)
                if coords.shape[0] > 0 and isinstance(coords.iloc[0, 1], str): 
                    coords = pd.read_csv(coord_path, header=0)
                else:
                    coords.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
            except Exception as e:
                if verbose: print(f"Error reading coords for {sample_name}: {e}")
                continue

            if coords is not None:
                if 'in_tissue' in coords.columns:
                    coords = coords[coords['in_tissue'] == 1]
                elif 'tissue' in coords.columns: # 某些旧格式可能使用 'tissue'
                    coords = coords[coords['tissue'] == 1]
                else:
                    if verbose: print(f"Warning: Columns 'in_tissue' or 'tissue' not found in {sample_name}. Assuming all spots are valid.")
            
            if coords is None or coords.empty:
                if verbose: print(f"Warning: {sample_name} has no valid in-tissue spots.")
                continue

            patch_sample_dir = cache_dir / sample_name
            total_in_tissue_spots += len(coords)
            
            missing_count = 0
            start_count = len(self.items)
            
            for _, row in coords.iterrows():
                barcode = row['barcode']
                
                pt_path = patch_sample_dir / f"{barcode}.pt"
                
                if not pt_path.exists():
                    if "-" in barcode:
                        alt_barcode = barcode.split("-")[0]
                    else:
                        alt_barcode = f"{barcode}-1"
                    
                    pt_path_alt = patch_sample_dir / f"{alt_barcode}.pt"
                    
                    if pt_path_alt.exists():
                        pt_path = pt_path_alt
                    else:
                        missing_count += 1
                        continue

                self.items.append(pt_path)
            
            if verbose:
                valid_count = len(self.items) - start_count
                if missing_count > 0:
                    print(f"  {sample_name}: Skipped {missing_count} missing/edge files.")
                print(f"  {sample_name}: {len(coords)} in-tissue spots, Loaded {valid_count} valid patches.")
            
        if verbose:
            print(f"--- Summary ---")
            print(f"Total in-tissue spots found: {total_in_tissue_spots}")
            print(f"Total usable patch count (after filtering): {len(self.items)}")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path = self.items[idx]
        
        try:
            img = torch.load(path)
            
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0).contiguous() 
                
                if img.max() <= 1.0:
                    img = (img * 255).to(torch.uint8)
                
                img_array = img.cpu().numpy()
                img = Image.fromarray(img_array) # 转换为 PIL Image

            if self.return_dual_view and self.transform:
                view1 = self.transform(img)
                view2 = self.transform(img)
                return view1, view2
            
            elif self.transform:
                img = self.transform(img)
                barcode = path.stem 
                return barcode, img
            
            else:
                barcode = path.stem
                return barcode, img
        
        except Exception as e:
            print(f"Error loading {path}: {e}")
            zero_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            if self.return_dual_view:
                return zero_tensor, zero_tensor
            else:
                return "error", zero_tensor

# class CachedPatchDataset(Dataset):
#     def __init__(self, raw_root, cache_dir, transform=None, only_sample=None, verbose=True):
#         self.items = []
#         self.transform = transform
#         raw_root = Path(raw_root).resolve()
#         cache_dir = Path(cache_dir).resolve()
        
#         spatial_dirs = list(raw_root.glob("*/spatial"))
        
#         if only_sample:
#             spatial_dirs = [d for d in spatial_dirs if d.parent.name == only_sample]
            
#         if verbose:
#             print(f"Found {len(spatial_dirs)} slices under {raw_root}")
            
#         total_spots = 0
        
#         for spatial_dir in spatial_dirs:
#             sample_name = spatial_dir.parent.name
            
#             coord_path = spatial_dir / "tissue_positions_list.csv"
#             if not coord_path.exists():
#                 coord_path = spatial_dir / "tissue_positions.csv"
            
#             if not coord_path.exists():
#                 if verbose: print(f"Warning: No coords found for {sample_name}")
#                 continue

#             try:
#                 coords = pd.read_csv(coord_path, header=None)
#                 if isinstance(coords.iloc[0, 1], str): 
#                     coords = pd.read_csv(coord_path, header=0)
#                 else:
#                     coords.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
#             except:
#                 if verbose: print(f"Error reading coords for {sample_name}")
#                 continue

#             if 'in_tissue' in coords.columns:
#                 coords = coords[coords['in_tissue'] == 1]
#             elif 'tissue' in coords.columns:
#                 coords = coords[coords['tissue'] == 1]

#             patch_sample_dir = cache_dir / sample_name
            
#             missing_count = 0
#             for _, row in coords.iterrows():
#                 barcode = row['barcode']
                
#                 pt_path = patch_sample_dir / f"{barcode}.pt"
                
#                 if not pt_path.exists():
#                     if "-" in barcode:
#                         alt_barcode = barcode.split("-")[0]
#                     else:
#                         alt_barcode = f"{barcode}-1"
                    
#                     pt_path_alt = patch_sample_dir / f"{alt_barcode}.pt"
                    
#                     if pt_path_alt.exists():
#                         pt_path = pt_path_alt
#                     else:
#                         missing_count += 1
#                         continue

#                 self.items.append(pt_path)
            
#             if verbose:
#                 if missing_count > 0:
#                     print(f"  {sample_name}: Skipped {missing_count} missing/edge files.")
#                 print(f"  {sample_name}: Loaded {len(self.items) - total_spots} valid patches.")
#             total_spots = len(self.items)
            
#         if verbose:
#             print(f" Total usable patch count: {len(self.items)}")
    
#     def __len__(self):
#         return len(self.items)
    
#     def __getitem__(self, idx):
#         path = self.items[idx]
#         try:
#             img = torch.load(path)
            
#             if isinstance(img, torch.Tensor):
#                 if img.dim() == 3 and img.shape[0] == 3:
#                     img = img.permute(1, 2, 0).contiguous() # 转换为 (H, W, C) 并确保内存连续
                
#                 if img.max() <= 1.0:
#                     img = (img * 255).to(torch.uint8)
                
#                 img_array = img.cpu().numpy()
#                 img = Image.fromarray(img_array)


#             if self.transform:
#                 img = self.transform(img) 
            
#             barcode = path.stem 
#             return barcode, img
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#             return "error", torch.zeros((3, 224, 224))


def load_finetuned_dinov2(model_path, device='cuda', use_projector=True):
    
    encoder = vit_giant2(patch_size=14)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith('pos_embed')}

    encoder.load_state_dict(model_state_dict, strict=False)
    encoder.eval().to(device)
    print(f"[INFO] Backbone loaded successfully on {device}")
    
    projector = None
    if use_projector and 'projector_state_dict' in checkpoint:
        projector = DINOHead(1536, 4096, 256).to(device).eval()
        projector.load_state_dict(checkpoint['projector_state_dict'])
        print(f"[INFO] Projection head loaded successfully")
    
    return encoder, projector

@torch.no_grad()
def extract_features(
    patch_dir="patch_cache_gpu",
    raw_dir="../RAW_SLICE/DLPFC",
    #raw_dir="../RAW_SLICE/test10x",
    model_path="results/dinov2_vitg14_spatial_full_dino_style.pt",  
    save_dir="features/spatial_patches_dinov2_spatial_full_dino",  
    batch_size=64,
    num_workers=8,
    use_projector=True
):
    patch_dir = Path(patch_dir).resolve()
    raw_dir = Path(raw_dir).resolve()
    print(f"Resolved patch_dir: {patch_dir}")
    print(f"Resolved raw_dir: {raw_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder, projector = load_finetuned_dinov2(model_path, device, use_projector=use_projector)
    
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    spatial_dirs = sorted(Path(raw_dir).glob("*/spatial"))
    # target_sample = "Mouse_Brain_Posterior" 
    # spatial_dirs = [d for d in spatial_dirs if d.parent.name == target_sample]
    
    if not spatial_dirs:
        print(f"No spatial data directories found under {raw_dir}. Please check your path.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    for spatial_dir in spatial_dirs:
        sample_name = spatial_dir.parent.name
        print(f"\n{'='*60}")
        print(f"Extracting features for sample: {sample_name}")
        print(f"{'='*60}")
        
        dataset = CachedPatchDataset(
            raw_root=raw_dir,
            cache_dir=patch_dir,
            transform=inference_transform, 
            only_sample=sample_name,
            verbose=True
        )
        
        print(f"Loaded {len(dataset)} valid patches for sample {sample_name}")
        
        if len(dataset) == 0:
            print(f" No valid patch for sample {sample_name}, skipping.")
            continue
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        )
        
        all_features = []
        all_barcodes = []
        
        for barcodes, patches in tqdm(dataloader, ncols=100, desc="Extracting"):
            patches = patches.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda'):
                all_tokens = encoder(patches) 
                                
                cls_token_feature = all_tokens 
                
                
                if cls_token_feature.dim() != 2 or cls_token_feature.shape[1] != 1536:
                    print(f"\n[FATAL ERROR] Unexpected CLS Token shape: {cls_token_feature.shape}")
                    raise RuntimeError("CLS Token dimension mismatch. Check ViT output or model configuration.")


                if projector is not None:
                    feats = projector(cls_token_feature)  # (B, proj_dim)
                else:
                    feats = F.normalize(cls_token_feature, p=2, dim=-1) # (B, D_model)
            
            feats = feats.cpu() 
            feats = feats.unsqueeze(1)  # (B, 1, feat_dim)
            all_features.append(feats)
            all_barcodes.extend(barcodes)
        
        features_tensor = torch.cat(all_features, dim=0)  # (N, 1, feat_dim)
        
        save_sample_dir = Path(save_dir)
        save_sample_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "barcodes": all_barcodes,
            "features": features_tensor
        }, save_sample_dir / f"{sample_name}.pt")
        
        np.save(save_sample_dir / f"{sample_name}.npy",
                np.array({"barcodes": all_barcodes, "features": features_tensor.numpy()}, dtype=object),
                allow_pickle=True)

        feat_dim = features_tensor.shape[-1]
        model_type = "with projector (256D)" if projector is not None else "backbone only (1536D)"
        print(f" Saved {sample_name}: {features_tensor.shape}")
        print(f"  - Barcodes: {len(all_barcodes)}")
        print(f"  - Feature dim: {feat_dim} (DINOv2 ViT-G/14 {model_type})")


if __name__ == "__main__":
    extract_features()