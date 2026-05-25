# Trustify: Fuzzy Logic-Based Hybrid Framework for Detecting Fake News

This repository is a complete implementation of the research paper **"A novel fuzzy logic-based hybrid framework for detecting fake news"** 

## Overview

**Trustify** is a multimodal fake news detection system that combines:
- **Text Processing**: LSTM networks for sequential text analysis
- **Image Processing**: ResNet50 + GNN + Transformer for visual feature extraction
- **Fuzzy Logic Decision Layer**: Interpretable reasoning for final classification
- **Text-Image Similarity Module**: Semantic alignment verification

## Key Features

✅ End-to-end pipeline: crawling → preprocessing → training → evaluation  
✅ Multimodal fusion with explainability  
✅ Supports 3 benchmark datasets: Twitter, BuzzFeed, PolitiFact  
✅ Achieves >96% accuracy on Twitter dataset  
✅ Ready for Colab/GPU training  
✅ Publication-quality visualizations  

## Performance Metrics (Paper Results)

| Dataset   | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Twitter   | 96.1%    | 98.4%     | 97.2%  | 97.8%    |
| BuzzFeed  | 95.9%    | 94.9%     | 92.2%  | 93.5%    |
| PolitiFact| 92.2%    | 94.9%     | 92.3%  | 93.9%    |

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA 12.1 (for GPU, optional but recommended)
- 8GB RAM minimum (16GB+ for comfortable training)

### Installation

```bash
# Clone repository
git clone https://github.com/fatimasood/NeuroVerify.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 1. Download/Prepare Datasets

The paper uses publicly available datasets. Download from:

```bash
# Create data directories
mkdir -p data/{twitter,buzzfeed,politifact}/{raw,processed}

# Twitter dataset
# Download from: https://www.kaggle.com/datasets/sudishsharma/twitter-fake-news-dataset

# BuzzFeed dataset
# Download from: https://www.kaggle.com/datasets/rmisra/buzzfeed-articles-fake-real-news

# PolitiFact dataset
# Download from: https://www.kaggle.com/datasets/rushi883/politifact-fake-real-dataset
```

### 2. Preprocess Data

```bash
python src/preprocessing/preprocess.py \
    --dataset twitter \
    --input_path data/twitter/raw \
    --output_path data/twitter/processed \
    --image_size 224
```

### 3. Train Model

```bash
python src/training/trainer.py \
    --dataset twitter \
    --epochs 20 \
    --batch_size 20 \
    --learning_rate 0.005 \
    --device cuda:0 \
    --output_dir checkpoints/
```

### 4. Evaluate Model

```bash
python src/evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset twitter \
    --batch_size 32 \
    --output_dir results/
```

### 5. Generate Visualizations

```bash
python src/utils/visualization.py \
    --results_dir results/ \
    --output_dir plots/
```

## Training on Google Colab

Open `notebooks/colab_train.ipynb` and:

1. Click "Open in Colab"
2. Run cells sequentially
3. GPU will be automatically enabled
4. Results and plots saved to Colab environment

```python
# In Colab:
!git clone https://github.com/fatimasood/NeuroVerify.git
!pip install -r requirements.txt
# Then follow notebook cells
```

