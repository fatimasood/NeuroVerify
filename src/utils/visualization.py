"""
Visualization utilities for Trustify framework
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import networkx as nx
import torch


def plot_training_history(history, save_path='outputs/plots/training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='s')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='^')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, dataset_name, save_path='outputs/plots/'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {dataset_name}')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_scores, dataset_name, save_path='outputs/plots/'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {dataset_name}')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/roc_curve_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc


def plot_fuzzy_membership_functions(fuzzy_system, save_path='outputs/plots/fuzzy_memberships.png'):
    """Plot fuzzy membership functions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(0, 1, 100)
    
    with torch.no_grad():
        for i, (ax, title) in enumerate(zip(axes, ['Text Confidence', 'Image Confidence'])):
            centers = fuzzy_system.membership.centers[i].cpu().numpy()
            sigmas = fuzzy_system.membership.sigmas[i].cpu().numpy()
            
            for j, (c, s) in enumerate(zip(centers, sigmas)):
                mf = np.exp(-((x - c) ** 2) / (2 * s ** 2))
                labels = ['Low', 'Medium', 'High']
                ax.plot(x, mf, label=f'{labels[j]}')
            
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Membership Degree')
            ax.set_title(f'Membership Functions - {title}')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rule_activation_strengths(rule_firings, dataset_name, save_path='outputs/plots/'):
    """Plot rule activation strengths"""
    rule_names = ['R1: T→Real, I→Real', 'R2: T→Fake, I→Real', 
                  'R3: T→Real, I→Fake', 'R4: T→Fake, I→Fake']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate average activation per rule
    avg_activations = rule_firings.mean(axis=0)
    
    bars = ax.bar(range(len(rule_names)), avg_activations, color=['green', 'orange', 'orange', 'red'])
    ax.set_xticks(range(len(rule_names)))
    ax.set_xticklabels(rule_names, rotation=45, ha='right')
    ax.set_ylabel('Average Activation Strength')
    ax.set_title(f'Fuzzy Rule Activation - {dataset_name}')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_activations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/rule_activation_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_df, save_path='outputs/plots/model_comparison.png'):
    """Plot comparison with existing models"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = results_df['Model'].values
    accuracy = results_df['Accuracy'].values
    
    bars = ax.bar(range(len(models)), accuracy, color='steelblue')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison on Fake News Detection')
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_architecture_diagram(save_path='outputs/plots/architecture.png'):
    """Plot Trustify architecture diagram"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define components
    components = {
        'Input': (1, 7),
        'Text': (2, 6.5),
        'Image': (2, 5.5),
        'LSTM': (3.5, 6.5),
        'GNN': (3.5, 5.5),
        'Transformer': (5, 6),
        'Fusion': (6.5, 6),
        'Fuzzy': (8, 6),
        'Output': (9, 6)
    }
    
    # Draw boxes
    for name, (x, y) in components.items():
        rect = plt.Rectangle((x-0.8, y-0.4), 1.6, 0.8, 
                              facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((1, 7), (2, 6.5)), ((1, 7), (2, 5.5)),
        ((2, 6.5), (3.5, 6.5)), ((2, 5.5), (3.5, 5.5)),
        ((3.5, 6.5), (5, 6)), ((3.5, 5.5), (5, 6)),
        ((5, 6), (6.5, 6)), ((6.5, 6), (8, 6)), ((8, 6), (9, 6))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    ax.set_title('Trustify Architecture', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_metrics(metrics_dict, dataset_name, save_path='outputs/plots/'):
    """Plot performance metrics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Metrics - {dataset_name}')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/metrics_{dataset_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()


def visualize_graph_network(node_features, adjacency_matrix, save_path='outputs/plots/graph_network.png'):
    """Visualize GNN graph structure"""
    G = nx.Graph()
    
    num_nodes = node_features.shape[0]
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges based on adjacency matrix
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adjacency_matrix[i, j] > 0.3:
                G.add_edge(i, j, weight=adjacency_matrix[i, j])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold',
            edge_color='gray', width=1.5, ax=ax)
    
    ax.set_title('GNN Graph Structure for Image Region Relationships')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()