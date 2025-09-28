"""
Training script for EfficientSAM Tiny & Small on COCO train2017 (full or subset)

Configurations:
- Loss: MSELoss (reconstruction)
- Optimizer: AdamW (β1=0.9, β2=0.999, weight_decay=0.1)
- Epochs: 40
- Image resolution: 1024 × 1024
- LR: 2e-4 with 10‑epoch linear warm-up + cosine decay
- Supports single‑GPU, DataParallel (multi‑GPU), or CPU fallback
- Train on a subset of COCO via --max_samples
"""
import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

# Build functions
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

class COCOSegmentationDataset(Dataset):
    def __init__(self, coco_root, ann_file, image_size, max_samples=None):
        self.coco = COCO(ann_file)
        self.ids = self.coco.getAnnIds()
        if max_samples is not None:
            self.ids = self.ids[:max_samples]
        self.image_root = os.path.join(coco_root, 'train2017')
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.loadAnns(ann_id)[0]
        # skip invalid
        if ann.get('iscrowd', 0) == 1 or 'segmentation' not in ann:
            return self.__getitem__((idx + 1) % len(self))

        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        mask = self.coco.annToMask(ann)
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask) / 255.0).float().unsqueeze(0)

        # sample two positive points
        ys, xs = torch.nonzero(mask[0], as_tuple=True)
        if ys.numel() < 2:
            points = torch.zeros((1,2,2), dtype=torch.float)
            labels = torch.zeros((1,2), dtype=torch.float)
        else:
            perm = torch.randperm(ys.size(0))[:2]
            pts = torch.stack([xs[perm], ys[perm]], dim=1).view(1,2,2).float()
            points = pts
            labels = torch.ones((1,2), dtype=torch.float)

        return img, points, labels, mask


def lr_lambda_fn(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))


def parse_args():
    p = argparse.ArgumentParser(description='Train EfficientSAM on COCO')
    p.add_argument('--coco_path',    type=str, default='data/coco2017')
    p.add_argument('--image_size',   type=int, default=1024)
    p.add_argument('--batch_size',   type=int, default=2)
    p.add_argument('--epochs',       type=int, default=40)
    p.add_argument('--warmup_epochs',type=int, default=10)
    p.add_argument('--lr',           type=float, default=2e-4)
    p.add_argument('--beta1',        type=float, default=0.9)
    p.add_argument('--beta2',        type=float, default=0.999)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--max_samples',  type=int, default=None,
                   help='limit number of annotations (subset of COCO)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    print(f'Using device: {device}, GPUs available: {num_gpus}')

    ann_file = os.path.join(args.coco_path, 'annotations', 'instances_train2017.json')
    dataset = COCOSegmentationDataset(
        args.coco_path, ann_file, args.image_size, args.max_samples
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    constructors = {
        'ti': build_efficient_sam_vitt,
        's':  build_efficient_sam_vits,
    }

    for key, build_fn in constructors.items():
        model = build_fn().to(device)
        if num_gpus > 1:
            model = nn.DataParallel(model)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr,
            betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ep: lr_lambda_fn(ep, args.warmup_epochs, args.epochs)
        )

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for imgs, points, labels, masks in loader:
                imgs, points, labels, masks = (
                    imgs.to(device), points.to(device),
                    labels.to(device), masks.to(device)
                )
                optimizer.zero_grad()
                pred_logits, _ = model(imgs, points, labels)
                pred_mask = torch.sigmoid(pred_logits[:,0])
                loss = criterion(pred_mask, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)
            print(f"[{key}] Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}")

        # save checkpoint
        ckpt = f"efficient_sam_vit{key}_final.pth"
        torch.save(model.module.state_dict() if num_gpus>1 else model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}\n")

if __name__ == '__main__':
    main()
