# LVIS
"""
EfficientSAM Evaluation on LVIS Dataset
======================================
This script evaluates the EfficientSAM model on LVIS dataset
and calculates metrics such as mIoU, AP, AR, etc.

Edit the configuration variables in the CONFIG section below to set paths and parameters.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import defaultdict
import json
import torchvision.transforms as T
from PIL import Image

# =====================================
# CONFIG - EDIT THESE PARAMETERS
# =====================================
# Path to your local dataset directory
DATASET_PATH = "/home/ritesh/Desktop/Rishabh/data/coco2017"

# Path to LVIS annotation file
LVIS_ANNOTATION_FILE = os.path.join(DATASET_PATH, "annotations", "/home/ritesh/Desktop/Rishabh/data/lvis/lvis_v1_val.json")

# Path to EfficientSAM weights directory
WEIGHTS_PATH = "/home/ritesh/Desktop/Rishabh/EfficientSAM/weights"

# EfficientSAM model type: "ti" for tiny, "s" for small
MODEL_TYPE = "s"  # Use 's' for better results, 'ti' for faster inference

# Type of prompt to use: "single_point", "two_points", or "box"
PROMPT_TYPE = "two_points"

# Number of images to evaluate (None for all images)
NUM_IMAGES = None

# Number of prompts to try per object (using the best result)
NUM_PROMPTS = 2

# Whether to generate visualizations
VISUALIZE = True

# Directory to save visualizations
VIS_DIR = "visualizations_efficientsam_lvis"

# Directory to save results
RESULTS_DIR = "results_efficientsam_lvis"

# Device to use (None for auto-detection, or specify like "cuda:0" or "cpu")
DEVICE = None
# =====================================

# Set device
if DEVICE is None:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(DEVICE)

print(f"Using device: {DEVICE}")

# Import EfficientSAM modules
try:
    from efficient_sam.build_efficient_sam import (
        build_efficient_sam_vitt,
        build_efficient_sam_vits,
    )
except ImportError:
    print("Error: Could not import efficient-sam. Please make sure it's installed.")
    print("Run: pip install git+https://github.com/yformer/EfficientSAM.git")
    sys.exit(1)

# Check if LVIS API is available
try:
    from lvis import LVIS, LVISEval
except ImportError:
    print("Error: Could not import LVIS API. Please install it with:")
    print("pip install lvis")
    sys.exit(1)

# Check if pycocotools is available
try:
    import pycocotools.mask as maskUtils
except ImportError:
    print("Error: Could not import pycocotools. Please install it with:")
    print("pip install pycocotools")
    sys.exit(1)

# Helper functions
def compute_iou(pred_mask, gt_mask):
    """Compute IoU between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return 0.0 if union == 0 else float(intersection / union)

def show_mask(mask, ax, random_color=False):
    """Display a mask on the given matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Display points (prompts) on the given matplotlib axis."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    
    # Add numbers to the points
    for i, (x, y) in enumerate(pos_points):
        ax.text(x, y+10, f"{i+1}", color='white', fontsize=12, 
                ha='center', va='center', fontweight='bold')

def decode_lvis_mask(ann, img_h, img_w):
    """Decode LVIS mask from segmentation format."""
    if isinstance(ann['segmentation'], dict):
        # RLE format
        return maskUtils.decode(ann['segmentation'])
    elif isinstance(ann['segmentation'], list):
        # Polygon format
        rles = maskUtils.frPyObjects(ann['segmentation'], img_h, img_w)
        rle = maskUtils.merge(rles)
        return maskUtils.decode(rle)
    else:
        raise ValueError(f"Unknown segmentation format: {type(ann['segmentation'])}")

def find_image_file(dataset_path, file_name):
    """Try to find the image file in different possible locations."""
    # List of possible paths to try
    paths_to_try = [
        os.path.join(dataset_path, file_name),                       # Direct in dataset root
        os.path.join(dataset_path, 'val2017', file_name),            # In val2017 subfolder
        os.path.join(dataset_path, 'coco2017', file_name),           # In coco2017 subfolder
        os.path.join(dataset_path, 'train2017', file_name),          # In train2017 subfolder
        os.path.join(dataset_path, 'images', 'val2017', file_name),  # In images/val2017
        os.path.join(dataset_path, 'images', file_name),             # In images folder
    ]
    
    # Try each path
    for path in paths_to_try:
        if os.path.exists(path):
            return path
    
    # If no path found, return None
    return None

def evaluate_efficientsam_on_lvis(
    lvis_annotation_file,
    dataset_path,
    model_type,
    weights_path,
    device,
    num_eval_images=None,
    prompt_type="two_points",
    num_prompts=2,
    visualize=False,
    vis_dir="visualizations_efficientsam_lvis"
):
    """Evaluate EfficientSAM on LVIS dataset."""
    print(f"Evaluating EfficientSAM on LVIS...")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Prompt type: {prompt_type}")
    print(f"Dataset path: {dataset_path}")
    print(f"LVIS annotation file: {lvis_annotation_file}")
    
    # Create visualization directory if needed
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Load LVIS dataset
    print("Loading LVIS annotations...")
    lvis = LVIS(lvis_annotation_file)
    
    # Get image IDs to evaluate
    image_ids = sorted(list(lvis.imgs.keys()))
    if num_eval_images is not None:
        image_ids = image_ids[:num_eval_images]
    print(f"Will evaluate on {len(image_ids)} images")
    
    # Print first few image info for debugging
    print("First few images info:")
    for i in range(min(3, len(image_ids))):
        img_id = image_ids[i]
        img_info = lvis.imgs[img_id]
        print(f"Image {i+1}: ID={img_id}, Info={img_info}")
        
        # Debug image path
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = os.path.basename(img_info['coco_url'])
        else:
            file_name = f"{img_id:012d}.jpg"
        
        file_path = find_image_file(dataset_path, file_name)
        print(f"  File name: {file_name}")
        print(f"  File path: {file_path}")
        print(f"  Exists: {file_path is not None}")
    
    # Load EfficientSAM model
    print(f"Loading EfficientSAM model ({model_type})...")
    if model_type == "ti":
        model = build_efficient_sam_vitt()
        # Check if weights file exists
        weights_file = os.path.join(weights_path, "efficient_sam_vitt.pt")
        if os.path.exists(weights_file):
            # Try to load weights with different approaches
            try:
                checkpoint = torch.load(weights_file, map_location=device)
                # Check if the checkpoint contains 'model' key
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                    print("Loaded model weights from checkpoint['model']")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights directly from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load weights from {weights_file}: {e}")
                print("Continuing with randomly initialized weights")
    elif model_type == "s":
        model = build_efficient_sam_vits()
        # Check if weights file exists
        weights_file = os.path.join(weights_path, "efficient_sam_vits.pt")
        if os.path.exists(weights_file):
            # Try to load weights with different approaches
            try:
                checkpoint = torch.load(weights_file, map_location=device)
                # Check if the checkpoint contains 'model' key
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                    print("Loaded model weights from checkpoint['model']")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights directly from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load weights from {weights_file}: {e}")
                print("Continuing with randomly initialized weights")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    # Define transform
    transform = T.ToTensor()
    
    # Run inference
    results = []  # LVIS results
    all_ious = []  # IoUs for all predictions
    skipped_images = 0
    skipped_annotations = 0
    
    start_time = time.time()
    
    # Show progress bar
    for img_idx, img_id in enumerate(tqdm(image_ids, desc="Processing images")):
        # Load image info
        img_info = lvis.imgs[img_id]
        
        # Determine file name
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        elif 'coco_url' in img_info:
            file_name = os.path.basename(img_info['coco_url'])
        else:
            file_name = f"{img_id:012d}.jpg"
        
        # Try to find the image file
        file_path = find_image_file(dataset_path, file_name)
        
        # Skip if file doesn't exist
        if not file_path:
            print(f"Warning: Image file not found: {file_name}")
            skipped_images += 1
            continue
        
        # Load image
        try:
            pil_img = Image.open(file_path).convert("RGB")
            img_tensor = transform(pil_img).to(device)
        except Exception as e:
            print(f"Warning: Failed to load image: {file_path}, error: {e}")
            skipped_images += 1
            continue
        
        # Also load the image for visualization if needed
        if visualize:
            image_np = np.array(pil_img)
        
        # Get annotations for this image
        ann_ids = lvis.get_ann_ids(img_ids=[img_id])
        ann_list = [lvis.anns[ann_id] for ann_id in ann_ids]
        
        if not ann_list:
            print(f"Warning: No annotations found for image ID {img_id}")
            skipped_images += 1
            continue
            
        # Process each annotation
        for ann_idx, ann in enumerate(ann_list):
            # Skip crowd annotations
            if ann.get('iscrowd', 0) == 1:
                skipped_annotations += 1
                continue
                
            cat_id = ann['category_id']
            
            # Skip annotations without segmentation
            if 'segmentation' not in ann or not ann['segmentation']:
                skipped_annotations += 1
                continue
                
            # Decode ground truth mask
            try:
                gt_mask = decode_lvis_mask(ann, img_info['height'], img_info['width'])
            except Exception as e:
                print(f"Warning: Failed to decode mask for annotation {ann_idx} in image {img_id}: {e}")
                skipped_annotations += 1
                continue
            
            # Find points inside the mask
            ys, xs = np.where(gt_mask == 1)
            
            # Skip if no points in mask
            if len(xs) == 0:
                skipped_annotations += 1
                continue
            
            # Track best result across multiple prompt attempts
            best_iou = -1
            best_mask = None
            best_mask_score = 0
            best_points = None
            best_labels = None
            
            # Try multiple prompts and keep the best one
            for prompt_attempt in range(num_prompts):
                # Generate prompts based on type
                if prompt_type == "single_point":
                    # Need at least one point
                    if len(xs) < 1:
                        continue
                    
                    # Pick a random point
                    idx = np.random.randint(0, len(xs))
                    input_point = np.array([[xs[idx], ys[idx]]])
                    input_label = np.array([1])  # 1 = foreground
                    
                    # Format for EfficientSAM
                    input_points = torch.tensor([[input_point]], dtype=torch.float, device=device)
                    input_labels = torch.tensor([[[1]]], dtype=torch.float, device=device)
                    
                elif prompt_type == "two_points":
                    # Need at least two points
                    if len(xs) < 2:
                        continue
                    
                    # Pick two random points
                    idxs = np.random.choice(len(xs), size=2, replace=False)
                    point1 = [xs[idxs[0]], ys[idxs[0]]]
                    point2 = [xs[idxs[1]], ys[idxs[1]]]
                    input_point = np.array([point1, point2])
                    input_label = np.array([1, 1])  # Both foreground
                    
                    # Format for EfficientSAM
                    input_points = torch.tensor([[input_point]], dtype=torch.float, device=device)
                    input_labels = torch.tensor([[[1, 1]]], dtype=torch.float, device=device)
                    
                elif prompt_type == "box":
                    # Need at least some points to create a box
                    if len(xs) < 1:
                        continue
                    
                    # Create box from mask points
                    x_min, y_min = np.min(xs), np.min(ys)
                    x_max, y_max = np.max(xs), np.max(ys)
                    
                    # For box prompts, we'll need to adapt this for EfficientSAM
                    # Currently, just skip box prompts as EfficientSAM might not handle them the same way
                    print("Box prompts are not yet implemented for EfficientSAM evaluation")
                    continue
                
                # Run EfficientSAM prediction
                try:
                    with torch.no_grad():
                        pred_logits, pred_iou = model(
                            img_tensor[None, ...],  # shape (1,3,H,W)
                            input_points,
                            input_labels
                        )
                        
                        # Sort by predicted IoU
                        sorted_ids = torch.argsort(pred_iou, dim=-1, descending=True)
                        pred_iou = torch.take_along_dim(pred_iou, sorted_ids, dim=2)
                        pred_logits = torch.take_along_dim(
                            pred_logits, sorted_ids[..., None, None], dim=2
                        )
                        
                        # Get the top-1 mask
                        mask_pred = (pred_logits[0, 0, 0] >= 0).cpu().numpy().astype(np.uint8)
                        mask_score = float(pred_iou[0, 0, 0].cpu().item())
                    
                    # Compute IoU
                    iou_val = compute_iou(mask_pred, gt_mask.astype(bool))
                    
                    # Update best result if this is better
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_mask = mask_pred
                        best_mask_score = mask_score
                        best_points = input_point
                        best_labels = input_label
                
                except Exception as e:
                    print(f"Error in EfficientSAM prediction for annotation {ann_idx} in image {img_id}: {e}")
                    continue
            
            # If we found any valid result
            if best_mask is not None:
                # Add to IoU list
                all_ious.append(best_iou)
                
                # Convert to RLE
                mask_fortran = np.asfortranarray(best_mask)
                pred_rle = maskUtils.encode(mask_fortran)
                pred_rle["counts"] = pred_rle["counts"].decode("ascii")
                
                # Add to results
                results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": pred_rle,
                    "score": float(best_mask_score)
                })
                
                # Visualize some results
                if visualize and (img_idx % 100 == 0) and (ann_idx < 3):
                    plt.figure(figsize=(10, 8))
                    plt.imshow(image_np)
                    show_mask(best_mask, plt.gca(), random_color=False)
                    show_points(best_points, best_labels, plt.gca())
                    plt.title(f"Image {img_id}, Object {ann_idx}\nIoU: {best_iou:.4f}, Score: {best_mask_score:.4f}")
                    plt.axis('off')
                    plt.tight_layout()
                    
                    vis_filename = f"{vis_dir}/img{img_id}_obj{ann_idx}.png"
                    plt.savefig(vis_filename)
                    plt.close()
            else:
                skipped_annotations += 1
            
            # Print occasional progress updates for large runs
            if len(all_ious) > 0 and len(all_ious) % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {len(all_ious)} objects in {elapsed:.2f}s "
                      f"({len(all_ious)/elapsed:.2f} objects/s)")
                print(f"Current Mean IoU: {np.mean(all_ious):.4f}")
    
    # Calculate evaluation time
    total_time = time.time() - start_time
    
    # Calculate mean IoU
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    print(f"Evaluation completed in {total_time:.2f} seconds")
    print(f"Processed {len(image_ids) - skipped_images} images")
    print(f"Generated {len(results)} mask predictions")
    print(f"Skipped {skipped_images} images and {skipped_annotations} annotations")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Evaluate using LVIS metrics
    metrics = {}
    if len(results) > 0:
        try:
            print("\nEvaluating using LVIS metrics...")
            
            # Save results to a temporary file for LVISEval
            import tempfile
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            json.dump(results, temp_file)
            temp_file.close()
            
            # Initialize LVIS evaluator
            lvis_results = LVISResults(lvis, temp_file.name)
            lvis_eval = LVISEval(lvis, lvis_results, 'segm')
            lvis_eval.evaluate()
            lvis_eval.accumulate()
            lvis_eval.summarize()
            
            # Extract metrics from LVIS evaluation
            metrics = {
                'AP': lvis_eval.results['AP'],
                'AP50': lvis_eval.results['AP50'],
                'AP75': lvis_eval.results['AP75'],
                'APs': lvis_eval.results['APs'],
                'APm': lvis_eval.results['APm'],
                'APl': lvis_eval.results['APl'],
                'APr': lvis_eval.results['APr'],
                'APc': lvis_eval.results['APc'],
                'APf': lvis_eval.results['APf']
            }
            
            # Clean up temp file
            os.unlink(temp_file.name)
        except Exception as e:
            print(f"Error during LVIS evaluation: {e}")
    else:
        print("No predictions to evaluate with LVIS metrics")
    
    # Create results dictionary
    evaluation_results = {
        'model_type': f"efficient_sam_{model_type}",
        'prompt_type': prompt_type,
        'num_prompts': num_prompts,
        'num_images': len(image_ids) - skipped_images,
        'num_predictions': len(results),
        'mean_iou': float(mean_iou),
        'evaluation_time': float(total_time),
        'lvis_metrics': metrics
    }
    
    return evaluation_results, results

# LVIS Results class needed for evaluation
class LVISResults:
    def __init__(self, lvis, results_file):
        """
        Load results file and create index
        """
        if isinstance(results_file, str):
            with open(results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = results_file
        
        self.lvis = lvis
        self._create_index()
        
    def _create_index(self):
        """
        Create index for results
        """
        self.img_ann = defaultdict(list)
        self.cat_img = defaultdict(list)
        
        for ann in self.results:
            self.img_ann[ann['image_id']].append(ann)
            self.cat_img[ann['category_id']].append(ann['image_id'])
        
        # Convert defaultdicts to regular dicts for serialization
        self.img_ann = dict(self.img_ann)
        self.cat_img = dict(self.cat_img)

def plot_results(results, results_dir):
    """Plot evaluation results."""
    if not results.get('lvis_metrics'):
        print("No LVIS metrics available to plot.")
        return
        
    metrics = results['lvis_metrics']
    
    plt.figure(figsize=(15, 6))
    
    # AP by IoU threshold
    plt.subplot(1, 3, 1)
    labels = ['AP', 'AP50', 'AP75']
    values = [metrics.get(k, 0) for k in labels]
    plt.bar(labels, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylim(0, 1.0)
    plt.title('Average Precision at Different IoU Thresholds')
    plt.ylabel('Average Precision')
    
    # AP by object size
    plt.subplot(1, 3, 2)
    labels = ['APs', 'APm', 'APl']
    values = [metrics.get(k, 0) for k in labels]
    plt.bar(labels, values, color=['#9b59b6', '#f39c12', '#1abc9c'])
    plt.ylim(0, 1.0)
    plt.title('Average Precision by Object Size')
    plt.ylabel('Average Precision')
    
    # AP by frequency
    plt.subplot(1, 3, 3)
    labels = ['APr', 'APc', 'APf']  # Rare, common, frequent
    values = [metrics.get(k, 0) for k in labels]
    plt.bar(labels, values, color=['#e74c3c', '#3498db', '#2ecc71'])
    plt.ylim(0, 1.0)
    plt.title('AP by Category Frequency')
    plt.ylabel('Average Precision')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"efficientsam_{results['model_type']}_{results['prompt_type']}_metrics_lvis.png"))
    plt.close()

def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run the evaluation
    print(f"Starting EfficientSAM evaluation with {PROMPT_TYPE} prompts on {MODEL_TYPE} model using LVIS dataset")
    
    evaluation_results, results = evaluate_efficientsam_on_lvis(
        lvis_annotation_file=LVIS_ANNOTATION_FILE,
        dataset_path=DATASET_PATH,
        model_type=MODEL_TYPE,
        weights_path=WEIGHTS_PATH,
        device=DEVICE,
        num_eval_images=NUM_IMAGES,
        prompt_type=PROMPT_TYPE,
        num_prompts=NUM_PROMPTS,
        visualize=VISUALIZE,
        vis_dir=VIS_DIR
    )
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, f"efficientsam_{MODEL_TYPE}_{PROMPT_TYPE}_results_lvis.json")
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Save detailed predictions
    predictions_file = os.path.join(RESULTS_DIR, f"efficientsam_{MODEL_TYPE}_{PROMPT_TYPE}_predictions_lvis.json")
    with open(predictions_file, 'w') as f:
        json.dump(results, f)
    
    print(f"Detailed predictions saved to: {predictions_file}")
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Model: EfficientSAM {evaluation_results['model_type']}")
    print(f"Prompt type: {evaluation_results['prompt_type']}")
    print(f"Number of images: {evaluation_results['num_images']}")
    print(f"Number of predictions: {evaluation_results['num_predictions']}")
    print(f"Mean IoU: {evaluation_results['mean_iou']:.4f}")
    print(f"Evaluation time: {evaluation_results['evaluation_time']:.2f} seconds")
    
    if evaluation_results.get('lvis_metrics'):
        print("\nLVIS Metrics:")
        for name, value in evaluation_results['lvis_metrics'].items():
            print(f"{name}: {value:.4f}")
        
        # Plot results
        plot_results(evaluation_results, RESULTS_DIR)
    else:
        print("\nNo LVIS metrics available.")

if __name__ == "__main__":
    main()