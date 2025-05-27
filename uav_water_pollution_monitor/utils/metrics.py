import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch # For IoU calculation if using PyTorch tensors

def calculate_iou(box1, box2, box_format="xyxy"):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Assumes boxes are in [x_min, y_min, x_max, y_max] format or [x_center, y_center, width, height].

    Args:
        box1 (list or np.array or torch.Tensor): First bounding box.
        box2 (list or np.array or torch.Tensor): Second bounding box.
        box_format (str): "xyxy" or "xywh".

    Returns:
        float: IoU value.
    """
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()

    if box_format == "xywh":
        # Convert to xyxy
        box1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2, box1[0] + box1[2]/2, box1[1] + box1[3]/2]
        box2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2, box2[0] + box2[2]/2, box2[1] + box2[3]/2]

    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_map(pred_boxes, true_boxes, pred_scores, pred_classes, true_classes, num_classes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP).
    This is a simplified mAP calculation. For rigorous mAP, refer to COCO evaluation scripts.

    Args:
        pred_boxes (list of lists): Predicted bounding boxes for each image.
                                    Each inner list contains [x_min, y_min, x_max, y_max].
        true_boxes (list of lists): Ground truth bounding boxes for each image.
        pred_scores (list of lists): Predicted confidence scores for each detection.
        pred_classes (list of lists): Predicted class labels for each detection.
        true_classes (list of lists): Ground truth class labels for each box.
        num_classes (int): Total number of classes.
        iou_threshold (float): IoU threshold to consider a detection as a True Positive.

    Returns:
        float: mAP value.
        dict: AP per class.
    """
    aps = {}
    for c in range(num_classes):
        # Get predictions and ground truths for this class
        class_preds = [] # (score, is_tp)
        num_gt_class = 0

        for i in range(len(pred_boxes)): # Iterate over images
            gt_image_boxes = [tb for idx, tb in enumerate(true_boxes[i]) if true_classes[i][idx] == c]
            pred_image_boxes = [pb for idx, pb in enumerate(pred_boxes[i]) if pred_classes[i][idx] == c]
            pred_image_scores = [ps for idx, ps in enumerate(pred_scores[i]) if pred_classes[i][idx] == c]

            num_gt_class += len(gt_image_boxes)
            detected_gt = [False] * len(gt_image_boxes)

            # Sort predictions by score
            if pred_image_boxes:
                sorted_indices = np.argsort(-np.array(pred_image_scores))
                pred_image_boxes = [pred_image_boxes[j] for j in sorted_indices]
                pred_image_scores_sorted = [pred_image_scores[j] for j in sorted_indices]


                for j, p_box in enumerate(pred_image_boxes):
                    score = pred_image_scores_sorted[j]
                    is_tp = False
                    if gt_image_boxes:
                        ious = [calculate_iou(p_box, gt_box) for gt_box in gt_image_boxes]
                        best_iou_idx = np.argmax(ious)
                        if ious[best_iou_idx] >= iou_threshold and not detected_gt[best_iou_idx]:
                            is_tp = True
                            detected_gt[best_iou_idx] = True
                    class_preds.append((score, is_tp))

        if not class_preds:
            aps[c] = 0.0
            continue

        class_preds.sort(key=lambda x: x[0], reverse=True)
        tp_arr = np.array([p[1] for p in class_preds], dtype=bool)
        fp_arr = ~tp_arr

        tp_cumsum = np.cumsum(tp_arr)
        fp_cumsum = np.cumsum(fp_arr)

        recalls = tp_cumsum / (num_gt_class + 1e-9) # Add epsilon to avoid division by zero
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

        # AP calculation using 11-point interpolation or COCO style (area under PR curve)
        # Simple average precision calculation for now
        ap = 0.0
        if num_gt_class > 0:
            # Use sklearn's average_precision_score for a more standard AP
            # This requires y_true (0 or 1 for each detection) and y_scores
            # For simplicity here, we'll use a basic PR curve area
            precisions = np.concatenate(([0.], precisions, [0.]))
            recalls = np.concatenate(([0.], recalls, [1.]))
            for k in range(len(precisions) - 1, 0, -1):
                precisions[k-1] = np.maximum(precisions[k-1], precisions[k])

            # Find indices where recall changes
            recall_change_indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[recall_change_indices + 1] - recalls[recall_change_indices]) * precisions[recall_change_indices + 1])
        aps[c] = ap

    map_value = np.mean(list(aps.values())) if aps else 0.0
    return map_value, aps

def get_classification_metrics(y_true, y_pred, y_pred_proba=None, average='weighted', class_names=None):
    """
    Calculates precision, recall, F1-score, accuracy, and confusion matrix.
    For multi-class, provide class_names for a more readable confusion matrix.
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)) if class_names else None)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm
    }

    # Per-class metrics if not using 'weighted' or 'macro' average
    if average is None and class_names:
        prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, name in enumerate(class_names):
            metrics[f"precision_{name}"] = prec_per_class[i]
            metrics[f"recall_{name}"] = rec_per_class[i]
            metrics[f"f1_score_{name}"] = f1_per_class[i]

    return metrics

if __name__ == '__main__':
    # Example Usage: IoU
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    print(f"IoU (xyxy): {calculate_iou(boxA, boxB)}")

    boxC_xywh = [5, 5, 10, 10] # center_x, center_y, width, height
    boxD_xywh = [10, 10, 10, 10]
    print(f"IoU (xywh): {calculate_iou(boxC_xywh, boxD_xywh, box_format='xywh')}")

    # Example Usage: Classification Metrics
    y_true_clf = [0, 1, 2, 0, 1, 2, 0, 0, 1]
    y_pred_clf = [0, 2, 1, 0, 0, 1, 2, 0, 1]
    class_names_clf = ['algae', 'trash', 'plastic']
    clf_metrics = get_classification_metrics(y_true_clf, y_pred_clf, average='weighted', class_names=class_names_clf)
    print(f"Classification Metrics (Weighted): {clf_metrics}")
    clf_metrics_per_class = get_classification_metrics(y_true_clf, y_pred_clf, average=None, class_names=class_names_clf)
    print(f"Classification Metrics (Per Class): {clf_metrics_per_class}")


    # Example Usage: mAP (Simplified)
    # Image 1
    pred_boxes_img1 = [[10, 10, 50, 50], [60, 60, 100, 100]]
    true_boxes_img1 = [[15, 15, 55, 55]]
    pred_scores_img1 = [0.9, 0.8]
    pred_classes_img1 = [0, 0]
    true_classes_img1 = [0]

    # Image 2
    pred_boxes_img2 = [[20, 20, 70, 70]]
    true_boxes_img2 = [[25, 25, 75, 75], [80, 80, 120, 120]]
    pred_scores_img2 = [0.95]
    pred_classes_img2 = [1]
    true_classes_img2 = [1, 1]

    all_pred_boxes = [pred_boxes_img1, pred_boxes_img2]
    all_true_boxes = [true_boxes_img1, true_boxes_img2]
    all_pred_scores = [pred_scores_img1, pred_scores_img2]
    all_pred_classes = [pred_classes_img1, pred_classes_img2]
    all_true_classes = [true_classes_img1, true_classes_img2]
    num_total_classes = 2 # Assuming classes 0 and 1

    map_val, aps_val = calculate_map(all_pred_boxes, all_true_boxes, all_pred_scores, all_pred_classes, all_true_classes, num_total_classes)
    print(f"Simplified mAP: {map_val}")
    print(f"AP per class: {aps_val}")