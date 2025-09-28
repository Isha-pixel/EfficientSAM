
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
from scipy import ndimage
import cv2

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
image_ids  = coco.getImgIds()  # full val2017

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
# 5) Strategic point selection functions
# -------------------------------------------------------------------------
def get_mask_boundary_points(mask, num_points=2):
    """Get points along the boundary of a mask."""
    # Create a boundary mask
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) - mask.astype(np.uint8)
    
    ys, xs = np.where(boundary > 0)
    if len(xs) < num_points:
        # Fallback to any points if boundary not clear
        ys, xs = np.where(mask > 0)
        
    if len(xs) < num_points:
        return None, None
        
    # Take points with maximum spacing along the boundary
    indices = np.linspace(0, len(xs) - 1, num_points).astype(int)
    return xs[indices], ys[indices]
    
def get_mask_centers(mask, num_points=2):
    """Get points from centers of the mask."""
    # Use distance transform to find central points
    dist_transform = ndimage.distance_transform_edt(mask)
    
    # Create a threshold to select central area
    threshold = 0.7 * dist_transform.max()
    central_region = dist_transform > threshold
    
    ys, xs = np.where(central_region)
    if len(xs) < num_points:
        # Fallback to any points if central region not clear
        ys, xs = np.where(mask > 0)
        
    if len(xs) < num_points:
        return None, None
        
    # Sample points to maximize distance between them
    if len(xs) == num_points:
        return xs, ys
        
    # Take random points from central region
    indices = np.random.choice(len(xs), num_points, replace=False)
    return xs[indices], ys[indices]

def analyze_mask_quality(mask):
    """Analyze mask quality and determine which refinement method to use."""
    if mask.sum() == 0:
        return "boundary"  # Default to boundary for empty masks
        
    # Count connected components
    labeled, num_components = ndimage.label(mask)
    if num_components > 1:
        return "center"  # Use center points to connect components
        
    # Get compactness (4π × Area / Perimeter²)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "boundary"
        
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    area = mask.sum()
    
    if perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter ** 2)
        if compactness < 0.6:  # Less compact shape
            return "boundary"  # Boundary points help define edges better
        else:
            return "center"  # Center points work better for compact shapes
    else:
        return "center"

# -------------------------------------------------------------------------
# 6) MULTI-ATTEMPT ADAPTIVE PROMPT INF LOOP
# -------------------------------------------------------------------------
def run_multi_adaptive_inference(coco_obj, image_ids, model, model_name, num_attempts=5):
    dt_list, iou_vals_top1 = [], []
    
    for img_id in tqdm(image_ids, desc=f"MultiAdaptiveInfer‑{model_name}"):
        # Load & prep image
        info = coco_obj.loadImgs(img_id)[0]
        img_path = os.path.join(coco_path, "val2017", info["file_name"])
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = val_transform(pil_img).to(device)
        
        # For each annotation
        for ann in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id)):
            if ann.get("iscrowd", 0) == 1 or "segmentation" not in ann:
                continue
                
            gt_mask = maskUtils.decode(coco_obj.annToRLE(ann))
            ys, xs = np.where(gt_mask == 1)
            
            if len(xs) < 2:
                continue  # need at least 2 points for a 'two-points' prompt
                
            cat_id = ann["category_id"]
            
            # For each attempt, we'll track the best result
            global_best_mask = None
            global_best_iou = -1
            global_best_score = None
            
            # Try multiple random starting points (like in original code)
            for attempt in range(num_attempts):
                # PHASE 1: Initial mask with random points
                # ----------------------------------------
                # Pick 2 random points from ground truth for initial mask
                idx1, idx2 = np.random.choice(len(xs), 2, replace=False)
                px1, py1 = xs[idx1], ys[idx1]
                px2, py2 = xs[idx2], ys[idx2]
                
                input_points = torch.tensor(
                    [[[[px1, py1], [px2, py2]]]],
                    dtype=torch.float, device=device)
                input_labels = torch.tensor(
                    [[[1, 1]]], dtype=torch.float, device=device)
                    
                # Run model forward - initial mask
                with torch.no_grad():
                    pred_logits, pred_iou = model(
                        img_tensor[None, ...],
                        input_points,
                        input_labels
                    )
                    # Sort by predicted iou
                    sorted_ids = torch.argsort(pred_iou, dim=-1, descending=True)
                    pred_iou = torch.take_along_dim(pred_iou, sorted_ids, dim=2)
                    pred_logits = torch.take_along_dim(
                        pred_logits, sorted_ids[..., None, None], dim=2
                    )
                    
                # Initial mask prediction
                initial_mask = (pred_logits[0, 0, 0] >= 0).cpu().numpy().astype(np.uint8)
                initial_iou_value = compute_iou(initial_mask, gt_mask)
                initial_score = float(pred_iou[0, 0, 0].cpu().item())
                
                # Save best results for this attempt (starting with initial)
                attempt_best_mask = initial_mask
                attempt_best_iou = initial_iou_value
                attempt_best_score = initial_score
                
                # PHASE 2: Adaptive Refinement
                # -----------------------------
                # Analyze mask quality to decide refinement strategy
                strategy = analyze_mask_quality(initial_mask)
                
                # Get refined points based on initial mask
                if strategy == "boundary":
                    refined_xs, refined_ys = get_mask_boundary_points(initial_mask)
                else:  # "center"
                    refined_xs, refined_ys = get_mask_centers(initial_mask)
                    
                # Skip refinement if we couldn't get points
                if refined_xs is None or refined_ys is None:
                    pass  # Just use the initial mask
                else:
                    # Create refined point tensor
                    refined_points = torch.tensor(
                        [[[[refined_xs[0], refined_ys[0]], [refined_xs[1], refined_ys[1]]]]],
                        dtype=torch.float, device=device)
                    
                    # Run model with refined points
                    with torch.no_grad():
                        refined_logits, refined_iou = model(
                            img_tensor[None, ...],
                            refined_points,
                            input_labels  # Same labels as before (all foreground)
                        )
                        # Sort by predicted iou
                        sorted_ids = torch.argsort(refined_iou, dim=-1, descending=True)
                        refined_iou = torch.take_along_dim(refined_iou, sorted_ids, dim=2)
                        refined_logits = torch.take_along_dim(
                            refined_logits, sorted_ids[..., None, None], dim=2
                        )
                    
                    # Get refined mask
                    refined_mask = (refined_logits[0, 0, 0] >= 0).cpu().numpy().astype(np.uint8)
                    refined_iou_value = compute_iou(refined_mask, gt_mask)
                    refined_score = float(refined_iou[0, 0, 0].cpu().item())
                    
                    # Update best if refined is better
                    if refined_iou_value > attempt_best_iou:
                        attempt_best_mask = refined_mask
                        attempt_best_iou = refined_iou_value
                        attempt_best_score = refined_score
                
                # Update global best if this attempt is better
                if attempt_best_iou > global_best_iou:
                    global_best_mask = attempt_best_mask
                    global_best_iou = attempt_best_iou
                    global_best_score = attempt_best_score
            
            # Record final best IoU across all attempts
            iou_vals_top1.append(global_best_iou)
            
            # Convert best_mask to RLE & store in dt_list
            mask_fortran = np.asfortranarray(global_best_mask)
            rle_pred = maskUtils.encode(mask_fortran)
            rle_pred["counts"] = rle_pred["counts"].decode("ascii")
            
            dt_list.append({
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": rle_pred,
                "score": global_best_score
            })
            
    mean_iou = np.mean(iou_vals_top1) if iou_vals_top1 else 0.0
    print(f"[{model_name}] Mean IoU (top-1) across {len(iou_vals_top1)} objects: {mean_iou:.4f}")
    return dt_list, mean_iou

# -------------------------------------------------------------------------
# 7) BASELINE (ORIGINAL) APPROACH
# -------------------------------------------------------------------------
def run_baseline_inference(coco_obj, image_ids, model, model_name, num_prompts=5):
    dt_list, iou_vals_top1 = [], []

    for img_id in tqdm(image_ids, desc=f"Baseline‑{model_name}"):
        # Load & prep image
        info = coco_obj.loadImgs(img_id)[0]
        img_path = os.path.join(coco_path, "val2017", info["file_name"])
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = val_transform(pil_img).to(device)

        # For each annotation
        for ann in coco_obj.loadAnns(coco_obj.getAnnIds(imgIds=img_id)):
            if ann.get("iscrowd", 0) == 1 or "segmentation" not in ann:
                continue

            gt_mask = maskUtils.decode(coco_obj.annToRLE(ann))
            ys, xs = np.where(gt_mask == 1)
            if len(xs) < 2:
                continue  # need at least 2 points for a 'two-points' prompt

            cat_id = ann["category_id"]

            # We'll track the best mask among NUM_PROMPTS attempts
            best_iou = -1
            best_mask = None
            best_score = None

            for attempt in range(num_prompts):
                # pick 2 random points
                idx1, idx2 = np.random.choice(len(xs), 2, replace=False)
                px1, py1 = xs[idx1], ys[idx1]
                px2, py2 = xs[idx2], ys[idx2]

                input_points = torch.tensor(
                    [[[[px1, py1], [px2, py2]]]],
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
                    sorted_ids = torch.argsort(pred_iou, dim=-1, descending=True)
                    pred_iou = torch.take_along_dim(pred_iou, sorted_ids, dim=2)
                    pred_logits = torch.take_along_dim(
                        pred_logits, sorted_ids[..., None, None], dim=2
                    )

                # for the top candidate c=0
                mask_pred = (pred_logits[0, 0, 0] >= 0).cpu().numpy().astype(np.uint8)
                iou_val = compute_iou(mask_pred, gt_mask)

                if iou_val > best_iou:
                    best_iou = iou_val
                    best_mask = mask_pred
                    best_score = float(pred_iou[0, 0, 0].cpu().item())

            # after trying N prompts
            iou_vals_top1.append(best_iou)

            # convert best_mask to RLE & store in dt_list
            mask_fortran = np.asfortranarray(best_mask)
            rle_pred = maskUtils.encode(mask_fortran)
            rle_pred["counts"] = rle_pred["counts"].decode("ascii")

            dt_list.append({
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": rle_pred,
                "score": best_score  # the predicted iou of the top candidate
            })

    mean_iou = np.mean(iou_vals_top1) if iou_vals_top1 else 0.0
    print(f"[{model_name}] Baseline Mean IoU (top-1) across {len(iou_vals_top1)} objects: {mean_iou:.4f}")
    return dt_list, mean_iou

# -------------------------------------------------------------------------
# 8) Main
# -------------------------------------------------------------------------
def main():
    results = {}
    num_prompts = 8
    
    # Option to use pre-computed baseline results instead of re-running
    use_precomputed_baselines = True
    precomputed_baseline_results = {
        "efficientsam_ti": {
            "mIoU": 0.757,
            "AP": 42.3
        },
        "efficientsam_s": {
            "mIoU": 0.77,
            "AP": 44.4
        }
    }
    
    # Run both baseline and adaptive for each model
    for name, model in models.items():
        print(f"\n=== EVALUATING MODEL: {name} ===")
        
        # Run baseline if not using precomputed results
        if not use_precomputed_baselines:
            print(f"\n- Running baseline (multi-prompt, {num_prompts} attempts)")
            baseline_dt_list, baseline_miou = run_baseline_inference(coco, image_ids, model, name, num_prompts)
            
            # COCOeval for baseline
            print("\nBaseline COCO metrics:")
            coco_dt = coco.loadRes(baseline_dt_list)
            coco_eval = COCOeval(coco, coco_dt, "segm")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Store baseline results
            results[name] = {
                'baseline_miou': baseline_miou,
                'baseline_AP': coco_eval.stats[0] * 100,  # AP @ IoU=0.5:0.95
                'baseline_AP50': coco_eval.stats[1] * 100,  # AP @ IoU=0.50
                'baseline_AP75': coco_eval.stats[2] * 100,  # AP @ IoU=0.75
                'baseline_AP_small': coco_eval.stats[3] * 100,  # AP small
                'baseline_AP_medium': coco_eval.stats[4] * 100,  # AP medium
                'baseline_AP_large': coco_eval.stats[5] * 100,  # AP large
                'baseline_AR1': coco_eval.stats[6] * 100,  # AR maxDets=1
                'baseline_AR10': coco_eval.stats[7] * 100,  # AR maxDets=10
                'baseline_AR100': coco_eval.stats[8] * 100,  # AR maxDets=100
            }
        else:
            # Use precomputed baseline results
            results[name] = {
                'baseline_miou': precomputed_baseline_results[name]["mIoU"],
                'baseline_AP': precomputed_baseline_results[name]["AP"]
            }
        
        # Run multi-adaptive approach (with same number of attempts)
        print(f"\n- Running adaptive prompt refinement")
        adaptive_dt_list, adaptive_miou = run_multi_adaptive_inference(coco, image_ids, model, name, num_prompts)
        
        # COCOeval for adaptive
        print("\nAdaptive Prompt Refinement COCO metrics:")
        coco_dt = coco.loadRes(adaptive_dt_list)
        coco_eval = COCOeval(coco, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Store adaptive results
        results[name]['adaptive_miou'] = adaptive_miou
        results[name]['improvement'] = (adaptive_miou - results[name]['baseline_miou']) / results[name]['baseline_miou'] * 100
        results[name]['adaptive_AP'] = coco_eval.stats[0] * 100  # AP @ IoU=0.5:0.95
        results[name]['adaptive_AP50'] = coco_eval.stats[1] * 100  # AP @ IoU=0.50
        results[name]['adaptive_AP75'] = coco_eval.stats[2] * 100  # AP @ IoU=0.75
        results[name]['adaptive_AP_small'] = coco_eval.stats[3] * 100  # AP small
        results[name]['adaptive_AP_medium'] = coco_eval.stats[4] * 100  # AP medium
        results[name]['adaptive_AP_large'] = coco_eval.stats[5] * 100  # AP large
        results[name]['adaptive_AR1'] = coco_eval.stats[6] * 100  # AR maxDets=1
        results[name]['adaptive_AR10'] = coco_eval.stats[7] * 100  # AR maxDets=10
        results[name]['adaptive_AR100'] = coco_eval.stats[8] * 100  # AR maxDets=100
    
    # Print final summary 
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        if use_precomputed_baselines:
            print(f"  Baseline mIoU:    {metrics['baseline_miou']:.4f}  (from previous run)")
            print(f"  Baseline AP:      {metrics['baseline_AP']:.2f}    (from previous run)")
        else:
            print(f"  Baseline mIoU:    {metrics['baseline_miou']:.4f}")
        
        print(f"  Adaptive mIoU:    {metrics['adaptive_miou']:.4f}")
        print(f"  mIoU Improvement: {metrics['improvement']:.2f}%")
        
        if not use_precomputed_baselines:
            print("\nBaseline COCO Metrics:")
            print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {metrics['baseline_AP']:.3f}")
            print(f"  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {metrics['baseline_AP50']:.3f}")
            print(f"  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {metrics['baseline_AP75']:.3f}")
            print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {metrics['baseline_AP_small']:.3f}")
            print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {metrics['baseline_AP_medium']:.3f}")
            print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {metrics['baseline_AP_large']:.3f}")
        
        print("\nAdaptive COCO Metrics:")
        print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {metrics['adaptive_AP']:.3f}")
        print(f"  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {metrics['adaptive_AP50']:.3f}")
        print(f"  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {metrics['adaptive_AP75']:.3f}")
        print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {metrics['adaptive_AP_small']:.3f}")
        print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {metrics['adaptive_AP_medium']:.3f}")
        print(f"  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {metrics['adaptive_AP_large']:.3f}")
        
        print("\nImprovements:")
        print(f"  AP Improvement:    {(metrics['adaptive_AP'] - metrics['baseline_AP']):.3f}")

if __name__=="__main__":
    main()