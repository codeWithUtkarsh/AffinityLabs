"""
Data loader module for Fashion Product Images dataset
@Author Utkarsh Sharma
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import zipfile
import shutil
from tqdm import tqdm
import config

class FashionDataLoader:
    """
    Handles loading and preprocessing of fashion dataset
    """

    def __init__(self, data_dir=None, use_subset=True, subset_size=2000):
        self.data_dir = data_dir or config.DATA_DIR
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.df = None

    def load_metadata(self):
        csv_path = self.data_dir / "styles.csv"

        if not csv_path.exists():
            print(f"CSV file not found at {csv_path}")
            print("Please download the dataset first using download_dataset.py")
            return None

        try:
            self.df = pd.read_csv(csv_path, on_bad_lines='skip')

            # Clean column names
            self.df.columns = self.df.columns.str.strip()

            # Handle missing values
            self.df = self.df.fillna({
                'gender': 'Unisex',
                'masterCategory': 'Other',
                'subCategory': 'Other',
                'articleType': 'Other',
                'baseColour': 'Multi',
                'season': 'All Season',
                'year': 2020,
                'usage': 'Casual',
                'productDisplayName': 'Unknown Product'
            })

            self.df['image_path'] = self.df['id'].apply(
                lambda x: str(self.data_dir / 'images' / f'{x}.jpg')
            )

            self.df['image_exists'] = self.df['image_path'].apply(os.path.exists)
            self.df = self.df[self.df['image_exists']].reset_index(drop=True)

            if self.use_subset and len(self.df) > self.subset_size:
                self.df = self._sample_diverse_subset()

            print(f"Loaded {len(self.df)} products")
            return self.df

        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None

    def _sample_diverse_subset(self):
        sampled_dfs = []

        categories = self.df['masterCategory'].unique()
        samples_per_category = max(1, self.subset_size // len(categories))

        for category in categories:
            category_df = self.df[self.df['masterCategory'] == category]
            n_samples = min(len(category_df), samples_per_category)
            sampled_dfs.append(category_df.sample(n=n_samples, random_state=42))

        result_df = pd.concat(sampled_dfs, ignore_index=True)

        if len(result_df) < self.subset_size:
            remaining = self.subset_size - len(result_df)
            additional = self.df[~self.df.index.isin(result_df.index)].sample(
                n=min(remaining, len(self.df) - len(result_df)),
                random_state=42
            )
            result_df = pd.concat([result_df, additional], ignore_index=True)

        return result_df.head(self.subset_size)

    def load_image(self, image_path, target_size=config.IMAGE_SIZE):
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def get_batch_images(self, indices, target_size=config.IMAGE_SIZE):
        images = []
        valid_indices = []

        for idx in indices:
            if idx < len(self.df):
                img_path = self.df.iloc[idx]['image_path']
                img = self.load_image(img_path, target_size)
                if img is not None:
                    images.append(img)
                    valid_indices.append(idx)

        return images, valid_indices

    def get_product_info(self, idx):
        if idx < len(self.df):
            return self.df.iloc[idx].to_dict()
        return None

    def filter_products(self, gender=None, category=None, color=None, season=None):
        filtered_df = self.df.copy()
        if gender and gender != "All":
            filtered_df = filtered_df[filtered_df['gender'] == gender]
        if category and category != "All":
            filtered_df = filtered_df[filtered_df['masterCategory'] == category]
        if color and color != "All":
            filtered_df = filtered_df[filtered_df['baseColour'] == color]
        if season and season != "All":
            filtered_df = filtered_df[filtered_df['season'] == season]
        return filtered_df

    def get_product_stats(self):
        if self.df is None:
            return None
        stats = {
            'total_products': len(self.df),
            'categories': self.df['masterCategory'].value_counts().to_dict(),
            'genders': self.df['gender'].value_counts().to_dict(),
            'colors': self.df['baseColour'].value_counts().head(10).to_dict(),
            'seasons': self.df['season'].value_counts().to_dict(),
            'article_types': self.df['articleType'].value_counts().head(10).to_dict()
        }
        return stats
