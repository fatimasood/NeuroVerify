"""
Twitter Fake News Dataset Crawler
Fetches tweets with images and labels
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

import tweepy
import requests
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwitterCrawler:
    """
    Crawler for Twitter fake news dataset.
    Note: For demonstration, uses pre-downloaded datasets.
    """
    
    def __init__(self, output_dir: str = "data/twitter/raw"):
        """
        Initialize crawler.
        
        Args:
            output_dir: Directory to save crawled data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # For demo: Using CSV format assuming data is pre-downloaded
        # In production, use tweepy with API credentials
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    def authenticate(self) -> Optional[tweepy.API]:
        """Authenticate with Twitter API (optional)."""
        if all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            try:
                auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
                auth.set_access_token(self.access_token, self.access_token_secret)
                api = tweepy.API(auth)
                logger.info("✓ Twitter API authentication successful")
                return api
            except Exception as e:
                logger.warning(f"Twitter API authentication failed: {e}")
        return None
    
    def download_image(self, url: str, filename: str) -> bool:
        """
        Download image from URL.
        
        Args:
            url: Image URL
            filename: Save filename
            
        Returns:
            Success status
        """
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img_path = self.output_dir / "images" / filename
                img_path.parent.mkdir(parents=True, exist_ok=True)
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            logger.warning(f"Failed to download image {url}: {e}")
        return False
    
    def load_csv_dataset(self, csv_path: str) -> List[Dict]:
        """
        Load pre-downloaded Twitter dataset from CSV.
        
        Expected columns: text, label (0=real, 1=fake), image_url
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} tweets from {csv_path}")
            
            data = []
            images_dir = self.output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing tweets"):
                item = {
                    'id': str(idx),
                    'text': row.get('text', ''),
                    'label': int(row.get('label', 0)),
                    'image_path': None,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'twitter'
                }
                
                # Download image if available
                if pd.notna(row.get('image_url')):
                    img_filename = f"twitter_{idx}.jpg"
                    if self.download_image(row['image_url'], img_filename):
                        item['image_path'] = str(images_dir / img_filename)
                
                data.append(item)
            
            return data
        except Exception as e:
            logger.error(f"Error loading CSV dataset: {e}")
            return []
    
    def create_sample_dataset(self, num_samples: int = 100) -> List[Dict]:
        """
        Create sample dataset for demo (without actual tweets).
        
        Args:
            num_samples: Number of samples to create
            
        Returns:
            List of sample data items
        """
        import random
        
        sample_texts = [
            "Breaking: Major political scandal uncovered",
            "New health study shows surprising results",
            "Celebrity reveals shocking truth",
            "Economic data shows positive growth",
            "Tech company announces new product",
        ]
        
        data = []
        for i in range(num_samples):
            item = {
                'id': f"twitter_{i}",
                'text': random.choice(sample_texts),
                'label': random.randint(0, 1),
                'image_path': None,
                'timestamp': datetime.now().isoformat(),
                'source': 'twitter'
            }
            data.append(item)
        
        logger.info(f"Created {num_samples} sample tweets")
        return data
    
    def save_data(self, data: List[Dict], filename: str = "tweets.json"):
        """Save data to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} items to {output_path}")
    
    def crawl(self, csv_path: Optional[str] = None, use_sample: bool = True):
        """
        Main crawl function.
        
        Args:
            csv_path: Path to pre-downloaded CSV dataset
            use_sample: If True, create sample data for demo
        """
        if use_sample:
            data = self.create_sample_dataset(num_samples=100)
        elif csv_path:
            data = self.load_csv_dataset(csv_path)
        else:
            logger.error("Provide either csv_path or set use_sample=True")
            return
        
        self.save_data(data)
        logger.info(f"✓ Twitter crawling complete: {len(data)} items")


if __name__ == "__main__":
    crawler = TwitterCrawler()
    
    # For demo: create sample data
    crawler.crawl(use_sample=True)
    
    # For production: provide CSV path
    crawler.crawl(csv_path="path/to/twitter_dataset.csv")