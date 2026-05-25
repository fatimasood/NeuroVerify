"""
BuzzFeed Fake News Dataset Crawler
Fetches articles with featured images
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
import random

import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuzzFeedCrawler:
    """Crawler for BuzzFeed fake news articles."""
    
    def __init__(self, output_dir: str = "data/buzzfeed/raw"):
        """Initialize crawler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_image(self, url: str, filename: str) -> bool:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img_path = self.output_dir / "images" / filename
                img_path.parent.mkdir(parents=True, exist_ok=True)
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            logger.warning(f"Failed to download image: {e}")
        return False
    
    def load_csv_dataset(self, csv_path: str) -> List[Dict]:
        """
        Load pre-downloaded BuzzFeed dataset from CSV.
        
        Expected columns: title, text, label, image_url
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} articles from {csv_path}")
            
            data = []
            images_dir = self.output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing articles"):
                text = f"{row.get('title', '')} {row.get('text', '')}"
                
                item = {
                    'id': f"buzzfeed_{idx}",
                    'title': row.get('title', ''),
                    'text': text.strip(),
                    'label': int(row.get('label', 0)),
                    'image_path': None,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'buzzfeed'
                }
                
                # Download image if available
                if pd.notna(row.get('image_url')):
                    img_filename = f"buzzfeed_{idx}.jpg"
                    if self.download_image(row['image_url'], img_filename):
                        item['image_path'] = str(images_dir / img_filename)
                
                data.append(item)
            
            return data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return []
    
    def create_sample_dataset(self, num_samples: int = 100) -> List[Dict]:
        """Create sample dataset for demo."""
        sample_titles = [
            "This One Weird Trick Will Blow Your Mind",
            "Celebrity Reveals Shocking Secret",
            "Scientists Make Breakthrough Discovery",
            "Local Mom Discovers Secret",
        ]
        
        sample_texts = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        ]
        
        data = []
        for i in range(num_samples):
            item = {
                'id': f"buzzfeed_{i}",
                'title': random.choice(sample_titles),
                'text': random.choice(sample_texts),
                'label': random.randint(0, 1),
                'image_path': None,
                'timestamp': datetime.now().isoformat(),
                'source': 'buzzfeed'
            }
            data.append(item)
        
        return data
    
    def save_data(self, data: List[Dict], filename: str = "articles.json"):
        """Save data to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} articles to {output_path}")
    
    def crawl(self, csv_path: Optional[str] = None, use_sample: bool = True):
        """Main crawl function."""
        if use_sample:
            data = self.create_sample_dataset(num_samples=100)
        elif csv_path:
            data = self.load_csv_dataset(csv_path)
        else:
            logger.error("Provide either csv_path or set use_sample=True")
            return
        
        self.save_data(data)
        logger.info(f"✓ BuzzFeed crawling complete: {len(data)} items")


if __name__ == "__main__":
    crawler = BuzzFeedCrawler()
    crawler.crawl(use_sample=True)