"""
Evaluation script for Trustify model
Computes metrics: Accuracy, Precision, Recall, F1-Score
"""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.trustify import TrustifyModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for Trustify model."""
    
    def __init__(self, model: TrustifyModel, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            model: Trustify model
            device: torch device
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_batch(self, batch: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions for batch.
        
        Args:
            batch: Data batch
            
        Returns:
            predictions, probabilities
        """
        text = batch['text'].to(self.device)
        images = batch['image'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(text, images)
            logits = outputs['logits']
            probs = outputs['probabilities']
            
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            probabilities = probs.cpu().numpy()
        
        return predictions, probabilities
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of metrics
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        logger.info("Evaluating model...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                predictions, probabilities = self.predict_batch(batch)
                labels = batch['label'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_probabilities.extend(probabilities[:, 1])
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1_score': f1_score(all_labels, all_predictions, zero_division=0),
            'auc_roc': roc_auc_score(all_labels, all_probabilities),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'classification_report': classification_report(
                all_labels, all_predictions, output_dict=True
            )
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict, dataset_name: str = 'Test'):
        """Print metrics summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"{dataset_name} Set Evaluation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        logger.info(f"{'='*60}\n")
    
    def plot_confusion_matrix(self, metrics: Dict, output_path: str):
        """Plot and save confusion matrix."""
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
    
    def plot_metrics_comparison(self, all_metrics: Dict, output_path: str):
        """Plot metrics comparison across datasets."""
        datasets = list(all_metrics.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric_name in enumerate(metrics_names):
            values = [all_metrics[ds][metric_name] for ds in datasets]
            
            axes[idx].bar(datasets, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[idx].set_ylabel(metric_name.capitalize())
            axes[idx].set_title(f'{metric_name.capitalize()} Comparison')
            axes[idx].set_ylim([0, 1.05])
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics comparison to {output_path}")


def load_checkpoint(checkpoint_path: str, device: torch.device) -> TrustifyModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = TrustifyModel(config)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='twitter')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Create evaluator
    evaluator = Evaluator(model, device)
    
    # Load test data (dummy for demo)
    from src.preprocessing.preprocess import FakeNewsDataset, ImagePreprocessor, TextPreprocessor
    import pickle
    
    # Load preprocessors
    with open(f"data/{args.dataset}/processed/text_processor.pkl", 'rb') as f:
        text_processor = pickle.load(f)
    
    image_processor = ImagePreprocessor()
    
    # Create test dataset
    test_path = f"data/{args.dataset}/processed/{args.dataset}_processed.json"
    test_dataset = FakeNewsDataset(test_path, text_processor, image_processor, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    metrics = evaluator.evaluate(test_loader)
    evaluator.print_metrics(metrics, f"{args.dataset.capitalize()} Test")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / f"{args.dataset}_metrics.json"
    with open(metrics_file, 'w') as f:
        # Convert numpy arrays to lists for JSON
        metrics_copy = metrics.copy()
        metrics_copy['confusion_matrix'] = np.array(metrics_copy['confusion_matrix']).tolist()
        json.dump(metrics_copy, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_file}")
    
    # Plot confusion matrix
    cm_path = output_dir / f"{args.dataset}_confusion_matrix.png"
    evaluator.plot_confusion_matrix(metrics, str(cm_path))
    
    logger.info(f"✓ Evaluation complete!")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    main()