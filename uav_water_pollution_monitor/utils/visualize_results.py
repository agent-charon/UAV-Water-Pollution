# Create this file: utils/visualize_results.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def draw_bounding_boxes(image_bgr, boxes, labels, scores=None, class_names=None, color_map=None, thickness=2):
    """
    Draws bounding boxes on an image.

    Args:
        image_bgr (np.array): The image (H, W, C) in BGR format (OpenCV default).
        boxes (list of lists/np.array): [[x_min, y_min, x_max, y_max], ...].
        labels (list/np.array): Class indices for each box.
        scores (list/np.array, optional): Confidence scores for each box.
        class_names (list, optional): List of class names. If None, labels are used directly.
        color_map (dict, optional): Dictionary mapping class_index to BGR color tuple.
                                    If None, random colors are generated.
        thickness (int): Thickness of the bounding box lines.

    Returns:
        np.array: Image with bounding boxes drawn.
    """
    vis_image = image_bgr.copy()
    if not class_names:
        class_names = [str(i) for i in range(max(labels) + 1 if labels else 0)]

    if color_map is None:
        # Generate random colors for each class if no map provided
        rng = np.random.default_rng(0) # Seed for reproducibility
        color_map = {i: tuple(rng.integers(0, 255, size=3).tolist()) for i in range(len(class_names))}

    for i, box in enumerate(boxes):
        label_idx = int(labels[i])
        color = color_map.get(label_idx, (0, 255, 0)) # Default to green if class not in map

        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, thickness)

        text = class_names[label_idx]
        if scores is not None and i < len(scores):
            text += f": {scores[i]:.2f}"

        # Put text above the box
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_image, (x_min, y_min - text_height - baseline), (x_min + text_width, y_min), color, -1) # Filled rect
        cv2.putText(vis_image, text, (x_min, y_min - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA) # Black text

    return vis_image

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues, output_path=None):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        cm (np.array): The confusion matrix.
        class_names (list): List of class names.
        title (str): Title of the plot.
        cmap (matplotlib.colors.Colormap): Colormap for the plot.
        output_path (str, optional): Path to save the plot. If None, shows the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if output_path:
        if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path) != '':
            os.makedirs(os.path.dirname(output_path))
        plt.savefig(output_path)
        print(f"Confusion matrix saved to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy'], title_prefix='', output_path=None):
    """
    Plots training and validation metrics over epochs.
    'history' should be a dictionary like:
    {
        'train_loss': [...], 'val_loss': [...],
        'train_accuracy': [...], 'val_accuracy': [...]
    }
    """
    num_metrics = len(metrics)
    plt.figure(figsize=(6 * num_metrics, 5))

    for i, metric_name in enumerate(metrics):
        plt.subplot(1, num_metrics, i + 1)
        if f'train_{metric_name}' in history:
            plt.plot(history[f'train_{metric_name}'], label=f'Train {metric_name.capitalize()}')
        if f'val_{metric_name}' in history:
            plt.plot(history[f'val_{metric_name}'], label=f'Validation {metric_name.capitalize()}')
        plt.title(f'{title_prefix} Training {metric_name.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if output_path:
        if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path) != '':
            os.makedirs(os.path.dirname(output_path))
        plt.savefig(output_path)
        print(f"Training history plot saved to {output_path}")
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Example Usage: draw_bounding_boxes
    sample_image = np.zeros((400, 600, 3), dtype=np.uint8) # Black image
    sample_image[:, :] = (200, 200, 200) # Light gray background

    boxes_ex = [[50, 50, 150, 150], [200, 100, 300, 250]]
    labels_ex = [0, 1] # Class indices
    scores_ex = [0.95, 0.88]
    class_names_ex = ['algae', 'trash']
    color_map_ex = {0: (0, 255, 0), 1: (0, 0, 255)} # Green for algae, Red for trash

    img_with_boxes = draw_bounding_boxes(sample_image, boxes_ex, labels_ex, scores_ex, class_names_ex, color_map_ex)
    cv2.imshow("Bounding Boxes Example", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Example Usage: plot_confusion_matrix
    from sklearn.metrics import confusion_matrix as sk_cm
    y_true_cm = [0, 1, 0, 1, 0, 1, 2, 2, 0]
    y_pred_cm = [0, 0, 0, 1, 0, 1, 1, 2, 0]
    class_names_cm = ['algae', 'trash', 'plastic']
    cm_data = sk_cm(y_true_cm, y_pred_cm, labels=np.arange(len(class_names_cm)))
    plot_confusion_matrix(cm_data, class_names_cm, title="Sample Confusion Matrix", output_path="results/sample_cm.png")

    # Example Usage: plot_training_history
    history_data = {
        'train_loss': [0.5, 0.3, 0.2, 0.15, 0.1],
        'val_loss': [0.6, 0.4, 0.3, 0.25, 0.22],
        'train_accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'val_accuracy': [0.65, 0.75, 0.82, 0.86, 0.88]
    }
    plot_training_history(history_data, metrics=['loss', 'accuracy'], title_prefix="MyModel", output_path="results/sample_training_history.png")