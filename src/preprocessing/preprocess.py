"""
Preprocessing pipeline for text and images
Text: Tokenization, embedding
Images: Resizing, normalization
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """Text preprocessing: tokenization, lowercasing, etc."""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 512):
        """
        Initialize text preprocessor.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum text length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = {}
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.stop_words = set(stopwords.words('english'))
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        word_freq = defaultdict(int)
        
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_freq[token] += 1
        
        # Keep top vocab_size words
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if len(self.word2idx) >= self.vocab_size:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        logger.info(f"Built vocabulary with {len(self.word2idx)} words")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        text = text.lower().strip()
        
        # Remove special characters
        import re
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        
        return tokens
    
    def encode(self, text: str) -> np.ndarray:
        """Convert text to indices."""
        tokens = self.tokenize(text)
        
        # Convert to indices
        indices = [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens[:self.max_length]
        ]
        
        # Pad to max_length
        if len(indices) < self.max_length:
            indices += [self.word2idx['<PAD>']] * (self.max_length - len(indices))
        
        return np.array(indices[:self.max_length], dtype=np.int64)


class ImagePreprocessor:
    """Image preprocessing: resizing, normalization."""
    
    def __init__(self, image_size: int = 224):
        """
        Initialize image preprocessor.
        
        Args:
            image_size: Target image size (square)
        """
        self.image_size = image_size
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process(self, image_path: str) -> Optional[torch.Tensor]:
        """Process image file."""
        try:
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                return self.get_placeholder()
            
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            return self.get_placeholder()
    
    def get_placeholder(self) -> torch.Tensor:
        """Return placeholder tensor if image unavailable."""
        return torch.zeros(3, self.image_size, self.image_size)


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection."""
    
    def __init__(
        self,
        data_path: str,
        text_processor: TextPreprocessor,
        image_processor: ImagePreprocessor,
        split: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON data file
            text_processor: Text preprocessor
            image_processor: Image preprocessor
            split: 'train', 'val', or 'test'
        """
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.split = split
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} items for split '{split}'")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single item."""
        item = self.data[idx]
        
        # Process text
        text_encoded = self.text_processor.encode(item['text'])
        
        # Process image
        image_tensor = self.image_processor.process(
            item.get('image_path', '')
        ) if item.get('image_path') else self.image_processor.get_placeholder()
        
        # Get label
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'text': torch.tensor(text_encoded, dtype=torch.long),
            'image': image_tensor,
            'label': label,
            'id': item.get('id', '')
        }


class PreprocessPipeline:
    """Complete preprocessing pipeline."""
    
    def __init__(self, config: Dict):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.text_processor = TextPreprocessor(
            vocab_size=config.get('vocab_size', 10000),
            max_length=config.get('max_text_length', 512)
        )
        self.image_processor = ImagePreprocessor(
            image_size=config.get('image_size', 224)
        )
    
    def preprocess(
        self,
        raw_data_path: str,
        output_dir: str,
        dataset_name: str = 'twitter'
    ):
        """
        Run full preprocessing pipeline.
        
        Args:
            raw_data_path: Path to raw JSON data
            output_dir: Output directory
            dataset_name: Name of dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        logger.info(f"Loading raw data from {raw_data_path}")
        with open(raw_data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Build vocabulary
        logger.info("Building vocabulary...")
        texts = [item['text'] for item in raw_data if 'text' in item]
        self.text_processor.build_vocab(texts)
        
        # Save preprocessors
        import pickle
        with open(output_dir / 'text_processor.pkl', 'wb') as f:
            pickle.dump(self.text_processor, f)
        
        # Create processed dataset
        processed_data = []
        for item in raw_data:
            processed_item = {
                'id': item.get('id', ''),
                'text': item.get('text', ''),
                'image_path': item.get('image_path', ''),
                'label': int(item.get('label', 0)),
                'source': item.get('source', 'unknown')
            }
            processed_data.append(processed_item)
        
        # Save processed data
        output_file = output_dir / f'{dataset_name}_processed.json'
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info(f"✓ Preprocessing complete. Saved to {output_file}")
        logger.info(f"  - Total items: {len(processed_data)}")
        logger.info(f"  - Vocabulary size: {len(self.text_processor.word2idx)}")


if __name__ == "__main__":
    # Example usage
    config = {
        'vocab_size': 10000,
        'max_text_length': 512,
        'image_size': 224
    }
    
    pipeline = PreprocessPipeline(config)
    
    # Preprocess each dataset
    for dataset in ['twitter', 'buzzfeed', 'politifact']:
        raw_path = f"data/{dataset}/raw/{dataset}_data.json"
        output_path = f"data/{dataset}/processed"
        
        if Path(raw_path).exists():
            pipeline.preprocess(raw_path, output_path, dataset)