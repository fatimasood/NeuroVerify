"""
PyTorch Dataset classes for multimodal fake news detection
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd


class MultimodalDataset(Dataset):
    """Dataset class for text + image fake news detection"""
    
    def __init__(self, dataframe, tokenizer, max_length=200, image_size=(224, 224), 
                 synthetic_images=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.synthetic_images = synthetic_images
        
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Text processing
        text = str(row['text']) if pd.notna(row['text']) else ""
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Image processing (synthetic or real)
        if self.synthetic_images:
            # Create synthetic random image for demonstration
            image = torch.randn(3, 224, 224)
        else:
            try:
                # Try to load real image (if available)
                image_path = row.get('image_path', None)
                if image_path and os.path.exists(image_path):
                    image = Image.open(image_path).convert('RGB')
                    image = self.image_transform(image)
                else:
                    image = torch.randn(3, 224, 224)
            except:
                image = torch.randn(3, 224, 224)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'image': image,
            'label': label,
            'text': text
        }


class TextDataset(Dataset):
    """Dataset for text-only processing"""
    
    def __init__(self, dataframe, tokenizer, max_length=200):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row['text']) if pd.notna(row['text']) else ""
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': label
        }