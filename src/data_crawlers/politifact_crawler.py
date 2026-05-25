"""
PolitiFact Fact-Check Dataset Crawler
Fetches political statements with fact-check labels
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
import random

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolitifactCrawler:
    """Crawler for PolitiFact fact-checking dataset."""
    
    def __init__(self, output_dir: str = "data/politifact/raw"):
        """Initialize crawler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_csv_dataset(self, csv_path: str) -> List[Dict]:
        """
        Load pre-downloaded PolitiFact dataset from CSV.
        
        Expected columns: statement, label, speaker, date
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} statements from {csv_path}")
            
            # Map ratings to binary labels: True/Mostly True/Pants on Fire -> Real/Fake
            label_mapping = {
                'true': 0, 'mostly-true': 0, 'half-true': 1,
                'mostly-false': 1, 'false': 1, 'pants-fire': 1
            }
            
            data = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing statements"):
                raw_label = str(row.get('label', 'false')).lower()
                label = label_mapping.get(raw_label, 1)
                
                item = {
                    'id': f"politifact_{idx}",
                    'statement': row.get('statement', ''),
                    'speaker': row.get('speaker', 'unknown'),
                    'label': label,
                    'image_path': None,
                    'timestamp': row.get('date', datetime.now().isoformat()),
                    'source': 'politifact',
                    'rating': raw_label
                }
                data.append(item)
            
            return data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return []
    
    def create_sample_dataset(self, num_samples: int = 100) -> List[Dict]:
        """Create sample dataset for demo."""
        sample_statements = [
            "The economy has grown by 3% this year",
            "Unemployment is at record highs",
            "Healthcare reforms have helped millions",
            "New infrastructure bill passed",
            "Climate data shows warming trend",
        ]
        
        sample_speakers = ["Senator Smith", "Rep. Johnson", "Expert Analyst", "News Report"]
        
        data = []
        for i in range(num_samples):
            item = {
                'id': f"politifact_{i}",
                'statement': random.choice(sample_statements),
                'speaker': random.choice(sample_speakers),
                'label': random.randint(0, 1),
                'image_path': None,
                'timestamp': datetime.now().isoformat(),
                'source': 'politifact',
                'rating': random.choice(['true', 'false'])
            }
            data.append(item)
        
        return data
    
    def save_data(self, data: List[Dict], filename: str = "statements.json"):
        """Save data to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} statements to {output_path}")
    
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
        logger.info(f"✓ PolitiFact crawling complete: {len(data)} items")


if __name__ == "__main__":
    crawler = PolitifactCrawler()
    crawler.crawl(use_sample=True)