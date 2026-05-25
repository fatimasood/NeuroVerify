"""
Visualization utilities for Trustify
Architecture diagrams and result plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_trustify_architecture(output_path: str = 'plots/architecture_diagram.png'):
    """
    Create detailed Trustify architecture diagram.
    
    Args:
        output_path: Save path for diagram
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    input_color = '#E8F4F8'
    text_color = '#FFE8CC'
    image_color = '#CCE8FF'
    fusion_color = '#E8CCFF'
    output_color = '#CCFFCC'
    
    # Title
    ax.text(5, 11.5, 'Trustify: Architecture Overview', 
            fontsize=20, fontweight='bold', ha='center')
    
    # ===== INPUT LAYER =====
    input_box = FancyBboxPatch((3.5, 10), 3, 0.8,
                               boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor=input_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 10.4, 'Input: Text + Image', fontsize=12, ha='center', fontweight='bold')
    
    # ===== TEXT BRANCH =====
    y_text_start = 9.2
    
    # Text input
    text_input = FancyBboxPatch((0.2, y_text_start), 2, 0.7,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=text_color, linewidth=1.5)
    ax.add_patch(text_input)
    ax.text(1.2, y_text_start + 0.35, 'Text Input', fontsize=10, ha='center')
    
    # Arrow down
    arrow = FancyArrowPatch((1.2, y_text_start), (1.2, y_text_start - 0.5),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # Tokenization
    token_box = FancyBboxPatch((0.2, y_text_start - 1.2), 2, 0.7,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor=text_color, linewidth=1.5)
    ax.add_patch(token_box)
    ax.text(1.2, y_text_start - 0.85, 'Tokenization', fontsize=10, ha='center')
    
    arrow = FancyArrowPatch((1.2, y_text_start - 1.2), (1.2, y_text_start - 1.7),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # Word Embedding
    embed_box = FancyBboxPatch((0.2, y_text_start - 2.4), 2, 0.7,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor=text_color, linewidth=1.5)
    ax.add_patch(embed_box)
    ax.text(1.2, y_text_start - 2.05, 'Word Embedding', fontsize=10, ha='center')
    
    arrow = FancyArrowPatch((1.2, y_text_start - 2.4), (1.2, y_text_start - 2.9),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # BiLSTM
    lstm_box = FancyBboxPatch((0.2, y_text_start - 3.6), 2, 0.7,
                              boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor=text_color, linewidth=1.5)
    ax.add_patch(lstm_box)
    ax.text(1.2, y_text_start - 3.25, 'BiLSTM', fontsize=10, ha='center', fontweight='bold')
    
    arrow = FancyArrowPatch((1.2, y_text_start - 3.6), (1.2, y_text_start - 4.1),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # Text Features
    text_feat = FancyBboxPatch((0.2, y_text_start - 4.8), 2, 0.7,
                               boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor=text_color, linewidth=1.5)
    ax.add_patch(text_feat)
    ax.text(1.2, y_text_start - 4.45, 'Text Features', fontsize=10, ha='center', fontweight='bold')
    
    # ===== IMAGE BRANCH =====
    y_image_start = 9.2
    
    # Image input
    image_input = FancyBboxPatch((7.8, y_image_start), 2, 0.7,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='black', facecolor=image_color, linewidth=1.5)
    ax.add_patch(image_input)
    ax.text(8.8, y_image_start + 0.35, 'Image Input', fontsize=10, ha='center')
    
    arrow = FancyArrowPatch((8.8, y_image_start), (8.8, y_image_start - 0.5),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # ResNet50
    resnet_box = FancyBboxPatch((7.8, y_image_start - 1.2), 2, 0.7,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=image_color, linewidth=1.5)
    ax.add_patch(resnet_box)
    ax.text(8.8, y_image_start - 0.85, 'ResNet50', fontsize=10, ha='center')
    
    arrow = FancyArrowPatch((8.8, y_image_start - 1.2), (8.8, y_image_start - 1.7),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # GNN
    gnn_box = FancyBboxPatch((7.8, y_image_start - 2.4), 2, 0.7,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=image_color, linewidth=1.5)
    ax.add_patch(gnn_box)
    ax.text(8.8, y_image_start - 2.05, 'Graph Neural Network', fontsize=9, ha='center')
    
    arrow = FancyArrowPatch((8.8, y_image_start - 2.4), (8.8, y_image_start - 2.9),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # Transformer
    transformer_box = FancyBboxPatch((7.8, y_image_start - 3.6), 2, 0.7,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='black', facecolor=image_color, linewidth=1.5)
    ax.add_patch(transformer_box)
    ax.text(8.8, y_image_start - 3.25, 'Transformer', fontsize=10, ha='center', fontweight='bold')
    
    arrow = FancyArrowPatch((8.8, y_image_start - 3.6), (8.8, y_image_start - 4.1),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # Image Features
    image_feat = FancyBboxPatch((7.8, y_image_start - 4.8), 2, 0.7,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=image_color, linewidth=1.5)
    ax.add_patch(image_feat)
    ax.text(8.8, y_image_start - 4.45, 'Image Features', fontsize=10, ha='center', fontweight='bold')
    
    # ===== FUSION LAYER =====
    # Arrows to fusion
    arrow = FancyArrowPatch((2.2, y_text_start - 4.45), (4, 3.5),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='purple')
    ax.add_patch(arrow)
    
    arrow = FancyArrowPatch((7.8, y_image_start - 4.45), (6, 3.5),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='purple')
    ax.add_patch(arrow)
    
    # Similarity Module
    similarity_box = FancyBboxPatch((3, 2.8), 4, 0.6,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='black', facecolor=fusion_color, linewidth=1.5)
    ax.add_patch(similarity_box)
    ax.text(5, 3.1, 'Text-Image Similarity Module', fontsize=11, ha='center', fontweight='bold')
    
    arrow = FancyArrowPatch((5, 2.8), (5, 2.3),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # Fuzzy Logic Layer
    fuzzy_box = FancyBboxPatch((2.5, 1.3), 5, 1,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=fusion_color, linewidth=2)
    ax.add_patch(fuzzy_box)
    ax.text(5, 2.05, 'Fuzzy Logic Decision Layer', fontsize=12, ha='center', fontweight='bold')
    ax.text(5, 1.65, 'Fuzzification → Rule Base → Inference → Defuzzification', 
            fontsize=9, ha='center', style='italic')
    
    arrow = FancyArrowPatch((5, 1.3), (5, 0.8),
                           arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow)
    
    # ===== OUTPUT LAYER =====
    output_box = FancyBboxPatch((3.5, 0), 3, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=output_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 0.4, 'Output: Real/Fake (with Confidence)', 
            fontsize=12, ha='center', fontweight='bold')
    
    # Legend
    legend_y = 11
    ax.text(0.2, legend_y, 'Legend:', fontsize=10, fontweight='bold')
    
    legend_items = [
        (input_color, 'Input/Output'),
        (text_color, 'Text Processing'),
        (image_color, 'Image Processing'),
        (fusion_color, 'Fusion'),
    ]
    
    for idx, (color, label) in enumerate(legend_items):
        y = legend_y - (idx + 1) * 0.35
        rect = mpatches.Rectangle((0.2, y - 0.12), 0.2, 0.2, 
                                  facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.5, y, label, fontsize=9, va='center')
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved architecture diagram to {output_path}")


def plot_model_comparison(all_results: dict, output_path: str = 'plots/model_comparison.png'):
    """
    Plot comparison of different models.
    
    Args:
        all_results: Dictionary with model results
        output_path: Save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = [all_results[model][metric] for model in models]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        bars = ax.bar(range(len(models)), values, color=colors[:len(models)])
        ax.set_ylabel(metric.replace('_', ' ').capitalize(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").capitalize()} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Trustify vs Other Models', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved model comparison to {output_path}")


if __name__ == "__main__":
    # Generate architecture diagram
    plot_trustify_architecture()
    
    # Example model comparison
    sample_results = {
        'LSTM (Text Only)': {'accuracy': 0.938, 'precision': 0.968, 'recall': 0.944, 'f1_score': 0.955},
        'ResNet+GNN (Image Only)': {'accuracy': 0.965, 'precision': 0.987, 'recall': 0.960, 'f1_score': 0.973},
        'Trustify (Hybrid)': {'accuracy': 0.961, 'precision': 0.984, 'recall': 0.972, 'f1_score': 0.978},
    }
    
    plot_model_comparison(sample_results)
    logger.info("✓ Visualization complete!")