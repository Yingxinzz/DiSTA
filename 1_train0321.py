import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import warnings
from dinov2.models.vision_transformer import vit_giant2


class DINOHead(nn.Module):
    """适配 CLS token 的 projection head"""
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
        return nn.functional.normalize(x, dim=-1)


class CachedPatchDataset(Dataset):
    """
    从patch cache加载空间转录组patches
    支持多个样本目录
    """
    def __init__(self, patch_dirs, spatial_dirs, transform=None, return_dual_view=True):
        self.samples = []
        self.transform = transform
        self.return_dual_view = return_dual_view
        
        for patch_dir, spatial_dir in zip(patch_dirs, spatial_dirs):
            coords_path = Path(spatial_dir) / "tissue_positions_list.csv"
            if not coords_path.exists():
                print(f"  Missing: {coords_path}")
                continue
            
            # 读取坐标文件
            coords = pd.read_csv(coords_path, sep="," if coords_path.suffix == ".csv" else None)
            if coords.shape[1] == 6:
                coords.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
            
            # 只保留在组织上的spots
            coords = coords[coords['in_tissue'] == 1]
            
            # 收集所有patch文件
            for _, row in coords.iterrows():
                barcode = row['barcode']
                pt_path = Path(patch_dir) / f"{barcode}.pt"
                if pt_path.exists():
                    self.samples.append(pt_path)
        
        print(f" Loaded {len(self.samples)} patches from {len(patch_dirs)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 加载cached patch tensor
        patch = torch.load(self.samples[idx])
        
        # 如果是tensor，转为PIL Image以便应用transforms
        if isinstance(patch, torch.Tensor):
            if patch.dim() == 3:
                if patch.shape[0] == 3:  # [C, H, W]
                    patch = patch.permute(1, 2, 0)  # -> [H, W, C]
                patch = patch.numpy()
            
            # 归一化到0-255
            if patch.max() <= 1.0:
                patch = (patch * 255).astype(np.uint8)
            
            patch = Image.fromarray(patch)
        
        if self.return_dual_view and self.transform is not None:
            # 返回两个增强视图
            view1 = self.transform(patch)
            view2 = self.transform(patch)
            return view1, view2
        elif self.transform is not None:
            return self.transform(patch), 0
        else:
            return patch, 0


def find_all_spatial_dirs(raw_roots):
    """
    从多个根目录中查找所有spatial文件夹
    """
    spatial_map = {}
    for root in raw_roots:
        for spatial_dir in Path(root).glob("*/spatial"):
            sample_id = spatial_dir.parent.name
            spatial_map[sample_id] = spatial_dir
    
    print(f" Found {len(spatial_map)} samples with spatial data")
    return spatial_map


class DINOv2SpatialFineTuner:
    """
    DINOv2微调器 - 空间转录组数据
    只训练 projection head，backbone 冻结
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()

        # 加载官方预训练权重
        print("Loading DINOv2 ViT-G/14 pretrained weights...")
        self.model = vit_giant2(patch_size=14)
        state_dict = torch.load("models/dinov2_vitg14_pretrain.pth", map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('pos_embed')}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print("Loaded weights:", msg)
        
        self.model = self.model.to(self.device)
        
        # 冻结 backbone
        for p in self.model.parameters():
            p.requires_grad = False
        print(" Backbone frozen")
        
        # Projection head
        in_dim = 1536
        self.projector = DINOHead(in_dim, config['proj_hidden_dim'], config['proj_dim']).to(self.device)
        
        print(" Projection head added (trainable)")
        
        # 只优化 projection head
        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=config['warmup_epochs']
        )
        
        self.current_epoch = 0
    
    def get_transforms(self):
        """
        空间转录组图像增强
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform
    
    def get_dataloader(self):
        """准备数据加载器"""
        transform = self.get_transforms()
        
        # 查找所有spatial目录
        spatial_map = find_all_spatial_dirs(self.config['raw_roots'])
        
        # 匹配patch_dirs和spatial_dirs
        patch_dirs = []
        spatial_dirs = []
        
        patch_root = Path(self.config['patch_root'])
        for sample_id, spatial_dir in spatial_map.items():
            patch_dir = patch_root / sample_id
            if patch_dir.exists():
                patch_dirs.append(patch_dir)
                spatial_dirs.append(spatial_dir)
        
        print(f" Matched {len(patch_dirs)} samples with patches")
        
        # 创建数据集
        dataset = CachedPatchDataset(
            patch_dirs=patch_dirs,
            spatial_dirs=spatial_dirs,
            transform=transform,
            return_dual_view=True
        )
        
        # 数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        return dataloader
    
    def compute_loss(self, feat1, feat2):
        """BYOL-style loss - 只在 projection head 上计算"""
        proj1 = self.projector(feat1)
        proj2 = self.projector(feat2)
        
        # 交叉预测
        loss1 = 2 - 2 * (proj1 * proj2.detach()).sum(dim=-1).mean()
        loss2 = 2 - 2 * (proj2 * proj1.detach()).sum(dim=-1).mean()
        
        return (loss1 + loss2) / 2
    
    def train_epoch(self, dataloader):
        self.model.eval()  # backbone 始终 eval
        self.projector.train()
        
        total_loss = 0
        progress_bar = tqdm(dataloader, 
                           desc=f"Epoch {self.current_epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (view1, view2) in enumerate(progress_bar):
            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)
            
            # 提取特征（无梯度）
            with torch.no_grad():
                feat1 = self.model(view1)
                feat2 = self.model(view2)
            
            # 只计算 projector 的梯度
            with autocast():
                loss = self.compute_loss(feat1, feat2)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.projector.parameters(),
                max_norm=3.0
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, save_path, is_best=False):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        print(f" Checkpoint saved: {save_path}")
        
        if is_best:
            best_path = save_path.parent / f"{save_path.stem}_best.pt"
            torch.save(checkpoint, best_path)
            print(f" Best model saved: {best_path}")
    
    def train(self):
        dataloader = self.get_dataloader()
        
        print(f"\n{'='*70}")
        print(f"DINOv2 Fine-tuning on Spatial Transcriptomics Data")
        print(f"{'='*70}")
        print(f"  Model: DINOv2 ViT-G/14 (Backbone frozen)")
        print(f"  Trainable: Projection head only")
        print(f"  Patch root: {self.config['patch_root']}")
        print(f"  Output: {self.config['output_path']}")
        print(f"  Raw roots: {len(self.config['raw_roots'])} directories")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Total samples: {len(dataloader.dataset)}")
        print(f"  Steps per epoch: {len(dataloader)}")
        print(f"  Expected time: ~2-4 hours")
        print(f"  Device: {self.device}")
        print(f"{'='*70}\n")
        
        best_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            if epoch < self.config['warmup_epochs']:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            avg_loss = self.train_epoch(dataloader)
            
            print(f" Epoch [{epoch+1}/{self.config['epochs']}] "
                  f"Avg Loss: {avg_loss:.4f} "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(self.config['output_path'], is_best=True)
            
            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(self.config['output_path']).parent / \
                                f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
            
            torch.cuda.empty_cache()
        
        final_path = Path(self.config['output_path']).parent / "final_model.pt"
        self.save_checkpoint(final_path)
        
        print("\n" + "="*70)
        print(" Fine-tuning completed!")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Best model: {self.config['output_path']}")
        print("="*70)


def finetune_dinov2_spatial(
    patch_root="patch_cache_gpu",
    raw_roots=["../RAW_SLICE/colorectal_liver", "../RAW_SLICE/training", 
               "../RAW_SLICE/zefish", "../RAW_SLICE/Human_Colon"],
    output_path="results/dinov2_vitg14_spatial_finetuned_complete.pt",
    batch_size=32,
    epochs=10,
    proj_dim=256,
    proj_hidden_dim=4096,
    num_workers=8,
    lr=0.001,
    warmup_epochs=1
):
    """
    DINOv2微调 - 空间转录组数据
    - Backbone 冻结
    - 只训练 projection head
    - 预计耗时：2-4 小时
    """
    
    config = {
        'patch_root': patch_root,
        'raw_roots': raw_roots,
        'output_path': output_path,
        'batch_size': batch_size,
        'epochs': epochs,
        'proj_dim': proj_dim,
        'proj_hidden_dim': proj_hidden_dim,
        'num_workers': num_workers,
        'lr': lr,
        'min_lr': 1e-5,
        'weight_decay': 0.01,
        'warmup_epochs': warmup_epochs
    }
    
    finetuner = DINOv2SpatialFineTuner(config)
    finetuner.train()
    
    return finetuner


if __name__ == "__main__":
    finetune_dinov2_spatial()