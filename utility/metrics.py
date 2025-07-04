"""Metrics tracking utilities."""
import numpy as np
import torch

## Taken from https://github.com/moatifbutt/r2s100k/blob/main/utils/metrics.py

def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_iou(predictions, target, num_classes, labeled):
    predictions = predictions * labeled.long()
    intersection = predictions * (predictions == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_classes, max=num_classes, min=1)
    area_pred = torch.histc(predictions.float(), bins=num_classes, max=num_classes, min=1)
    area_lab = torch.histc(target.float(), bins=num_classes, max=num_classes, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metric(output, target, num_classes):
    _, predictions = torch.max(output.data, 1)
    predictions = predictions + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_classes)
    correct, num_labeled = batch_pix_accuracy(predictions, target, labeled)
    inter, union = batch_iou(predictions, target, num_classes, labeled)
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5)]

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ConfusionMatrix:
    """Computes confusion matrix for semantic segmentation."""
    
    def __init__(self, num_classes, ignore_index):
        """
        Initialize confusion matrix.
        
        Args:
            num_classes: Number of classes in the segmentation task
        """
        self.num_classes = num_classes
        self.mat = None
        self.ignore_index = ignore_index

    def update(self, pred, target):
        """
        Update confusion matrix.
        
        Args:
            pred: Predicted segmentation mask (B, H, W) or (B, C, H, W)
            target: Ground truth segmentation mask (B, H, W)
        """
        # Convert predictions to class indices if needed
        if pred.dim() == 4:
            # If predictions are class logits (B, C, H, W), convert to class indices
            pred = torch.argmax(pred, dim=1)
        
        # Convert tensors to numpy arrays
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Create or update confusion matrix
        if self.mat is None:
            self.mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        # Fill in confusion matrix
        mask = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        
        if mask.sum() > 0:
            self.mat += np.bincount(
                self.num_classes * target[mask].astype(int) + pred[mask],
                minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
    
    def reset(self):
        """Reset confusion matrix."""
        self.mat = None
    
    def compute(self):
        """
        Compute metrics from confusion matrix.
        
        Returns:
            Dictionary of metrics including:
            - pixel_accuracy: Overall pixel accuracy
            - class_accuracy: Per-class accuracy
            - mean_accuracy: Mean class accuracy
            - iou: Per-class IoU
            - mean_iou: Mean IoU
            - fw_iou: Frequency-weighted IoU
        """
        if self.mat is None:
            return {}
        
        # Calculate relevant metrics
        h = self.mat.astype(np.float64)

        # Accuracy: overall accuracy
        if self.ignore_index < self.num_classes and self.ignore_index >= 0:
            h = np.delete(self.mat, self.ignore_index, axis=0)
            h = np.delete(h, self.ignore_index, axis=1)
        acc = np.diag(h).sum() / h.sum()
        
        h = self.mat.astype(np.float64)
        # Accuracy per class and mean accuracy
        acc_per_class = np.diag(h) / h.sum(axis=1)
        acc_per_class = np.nan_to_num(acc_per_class)

        if self.ignore_index < self.num_classes and self.ignore_index >= 0:
            mean_acc = np.delete(acc_per_class, 0, axis=0).mean()
        else:
            mean_acc = acc_per_class.mean()

        
        # IoU per class and mean IoU
        iou = np.diag(h) / (h.sum(axis=1) + h.sum(axis=0) - np.diag(h))
        iou = np.nan_to_num(iou)
        if self.ignore_index < self.num_classes and self.ignore_index >= 0:
            mean_iou = np.delete(iou, 0, axis=0).mean()
        else:
            mean_iou = iou.mean()
        
        # Frequency-weighted IoU
        freq = h.sum(axis=1) / h.sum()
        fw_iou = (freq * iou).sum()
        
        # Return all metrics
        return {
            'pixel_accuracy': acc,
            'class_accuracy': acc_per_class,
            'mean_accuracy': mean_acc,
            'iou': iou,
            'mean_iou': mean_iou,
            'fw_iou': fw_iou
        }
    
    def get_iou_per_class(self):
        """
        Get IoU value for each class.
        
        Returns:
            Numpy array with IoU for each class
        """
        if self.mat is None:
            return np.zeros(self.num_classes)
        
        h = self.mat.astype(np.float64)
        iou = np.diag(h) / (h.sum(axis=1) + h.sum(axis=0) - np.diag(h))
        return np.nan_to_num(iou)
    
    def get_mean_iou(self):
        """
        Get mean IoU.
        
        Returns:
            Mean IoU value
        """
        return np.mean(self.get_iou_per_class())