import os
import zipfile
import torch
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as T
from tqdm import tqdm

from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vitt,
    build_efficient_sam_vits,
)


# -------------------------------------------------------------------------
# DEVICE SET‑UP
# -------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")
torch.backends.cudnn.benchmark = True

# -------------------------------------------------------------------------
# 1) SETUP
# -------------------------------------------------------------------------
weights_path = "/home/ritesh/Desktop/Rishabh/EfficientSAM/weights"
coco_path    = "/home/ritesh/Desktop/Rishabh/data/coco2017"
vits_zip     = os.path.join(weights_path, "efficient_sam_vits.pt.zip")
vits_pt      = os.path.join(weights_path, "efficient_sam_vits.pt")

if (not os.path.exists(vits_pt)) and os.path.exists(vits_zip):
    with zipfile.ZipFile(vits_zip, "r") as zip_ref:
        zip_ref.extractall(weights_path)

annotation_file = os.path.join(coco_path, "annotations", "instances_val2017.json")
coco       = COCO(annotation_file)
image_ids  = coco.getImgIds()

# -------------------------------------------------------------------------
# 2) LOAD EFFICIENTSAM MODELS → GPU
# -------------------------------------------------------------------------
models = {
    "efficientsam_ti": build_efficient_sam_vitt().to(device).eval(),
    "efficientsam_s":  build_efficient_sam_vits().to(device).eval(),
}

# -------------------------------------------------------------------------
# 3) TRANSFORM
# -------------------------------------------------------------------------
val_transform = T.ToTensor()

# -------------------------------------------------------------------------
# 4) IoU helper
# -------------------------------------------------------------------------
def compute_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union        = (pred_mask | gt_mask).sum()
    return 0.0 if union == 0 else float(intersection / union)

# -------------------------------------------------------------------------
# 5) MULTI‑PROMPT INF LOOP
# -------------------------------------------------------------------------
# We'll evaluate each model across the entire val set
NUM_PROMPTS = 5 

def run_inference_on_images(coco_obj, image_ids, model, model_name):
    dt_list, iou_vals_top1 = [], []

    for img_id in tqdm(image_ids, desc=f"Infer‑{model_name}"):
        # Load & prep image
        info       = coco_obj.loadImgs(img_id)[0]
        img_path   = os.path.join(coco_path, "val2017", info["file_name"])
        pil_img    = Image.open(img_path).convert("RGB")
        img_tensor = val_transform(pil_img).to(device)

        # For each annotation
        for ann in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id)):
            if ann.get("iscrowd",0)==1 or "segmentation" not in ann:
                continue

            gt_mask = maskUtils.decode(coco_obj.annToRLE(ann))
            ys, xs  = np.where(gt_mask == 1)
            if len(xs) < 2:
                continue  # need at least 2 points for a 'two-points' prompt

            cat_id       = ann["category_id"]

            
            best_iou     = -1
            best_mask    = None
            best_score   = None

            for attempt in range(NUM_PROMPTS):
                # pick 2 random points
                idx1, idx2 = np.random.choice(len(xs), 2, replace=False)
                px1, py1   = xs[idx1], ys[idx1]
                px2, py2   = xs[idx2], ys[idx2]

                input_points = torch.tensor(
                    [[[[px1, py1],[px2, py2]]]],
                    dtype=torch.float, device=device)
                input_labels = torch.tensor(
                    [[[1, 1]]], dtype=torch.float, device=device)

                # run model forward
                with torch.no_grad():
                    pred_logits, pred_iou = model(
                        img_tensor[None, ...],  # shape (1,3,H,W)
                        input_points,
                        input_labels
                    )
                    # sort by predicted iou
                    sorted_ids  = torch.argsort(pred_iou, dim=-1, descending=True)
                    pred_iou    = torch.take_along_dim(pred_iou, sorted_ids, dim=2)
                    pred_logits = torch.take_along_dim(
                        pred_logits, sorted_ids[...,None,None], dim=2
                    )

                # for the top candidate c=0
                mask_pred = (pred_logits[0, 0, 0] >= 0).cpu().numpy().astype(np.uint8)
                iou_val   = compute_iou(mask_pred, gt_mask)

                if iou_val > best_iou:
                    best_iou   = iou_val
                    best_mask  = mask_pred
                    best_score = float(pred_iou[0, 0, 0].cpu().item())

            
            iou_vals_top1.append(best_iou)

            # convert best_mask to RLE & store in dt_list
            mask_fortran         = np.asfortranarray(best_mask)
            rle_pred             = maskUtils.encode(mask_fortran)
            rle_pred["counts"]   = rle_pred["counts"].decode("ascii")

            dt_list.append({
                "image_id":    img_id,
                "category_id": cat_id,
                "segmentation": rle_pred,
                "score":       best_score  # the predicted iou of the top candidate
            })

    mean_iou = np.mean(iou_vals_top1) if iou_vals_top1 else 0.0
    print(f"[{model_name}] Mean IoU (top-1) across {len(iou_vals_top1)} objects: {mean_iou:.4f}")
    return dt_list

# -------------------------------------------------------------------------
# 6) Main
# -------------------------------------------------------------------------
def main():
    for name, model in models.items():
        print(f"\n=== EVALUATING MODEL: {name} ===")
        coco_dt_list = run_inference_on_images(coco, image_ids, model, name)
        if not coco_dt_list:
            print("No predictions made, skipping COCOeval.")
            continue

        # COCOeval
        coco_dt   = coco.loadRes(coco_dt_list)
        coco_eval = COCOeval(coco, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

if __name__=="__main__":
    main()
