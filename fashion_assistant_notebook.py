#!/usr/bin/env python
# coding: utf-8

"""
Fashion Shopping Assistant - Jupyter Notebook Version
This notebook provides an interactive fashion recommendation system
"""

# # Fashion Shopping Assistant - Notebook Version
#
# This notebook demonstrates a fashion recommendation system using deep learning and image similarity search.

# ## 1. Setup and Imports

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
import pickle

# Deep Learning
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Similarity Search
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ## 2. Configuration

class Config:
    """Configuration settings"""
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    IMAGES_DIR = DATA_DIR / "images"
    STYLES_CSV = DATA_DIR / "styles.csv"
    FEATURES_DIR = BASE_DIR / "features"

    # Model settings
    MODEL_NAME = "resnet50"
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    FEATURE_DIM = 2048

    # Search settings
    TOP_K = 10

    # Create directories
    for dir_path in [DATA_DIR, FEATURES_DIR]:
        dir_path.mkdir(exist_ok=True)

config = Config()

# ## 3. Data Loading

class FashionDataLoader:
    """Handle fashion dataset loading"""

    def __init__(self, config, use_subset=True, subset_size=1000):
        self.config = config
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.df = None

    def load_metadata(self):
        """Load product metadata"""
        if not self.config.STYLES_CSV.exists():
            print(f"Dataset not found at {self.config.STYLES_CSV}")
            print("Please download the dataset first!")
            return None

        self.df = pd.read_csv(self.config.STYLES_CSV, on_bad_lines='skip')
        self.df.columns = self.df.columns.str.strip()

        # Fill missing values
        self.df = self.df.fillna({
            'gender': 'Unisex',
            'masterCategory': 'Other',
            'baseColour': 'Multi',
            'season': 'All Season',
            'productDisplayName': 'Unknown Product'
        })

        # Add image paths
        self.df['image_path'] = self.df['id'].apply(
            lambda x: str(self.config.IMAGES_DIR / f'{x}.jpg')
        )

        # Filter to existing images
        self.df['image_exists'] = self.df['image_path'].apply(os.path.exists)
        self.df = self.df[self.df['image_exists']].reset_index(drop=True)

        if self.use_subset and len(self.df) > self.subset_size:
            self.df = self.df.sample(n=self.subset_size, random_state=42)

        print(f"Loaded {len(self.df)} products")
        return self.df

    def load_image(self, image_path):
        """Load and preprocess image"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img.resize(self.config.IMAGE_SIZE, Image.Resampling.LANCZOS)
        except:
            return None

# Load data
data_loader = FashionDataLoader(config, use_subset=True, subset_size=1000)
df = data_loader.load_metadata()

if df is not None:
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nSample data:")
    display(df.head())

# ## 4. Data Exploration

if df is not None:
    # Category distribution
    fig = px.pie(
        values=df['masterCategory'].value_counts().values,
        names=df['masterCategory'].value_counts().index,
        title="Product Category Distribution"
    )
    fig.show()

    # Gender distribution
    fig = px.bar(
        x=df['gender'].value_counts().index,
        y=df['gender'].value_counts().values,
        title="Gender Distribution",
        labels={'x': 'Gender', 'y': 'Count'}
    )
    fig.show()

    # Color distribution (top 10)
    top_colors = df['baseColour'].value_counts().head(10)
    fig = px.bar(
        x=top_colors.values,
        y=top_colors.index,
        orientation='h',
        title="Top 10 Colors",
        labels={'x': 'Count', 'y': 'Color'}
    )
    fig.show()

# ## 5. Feature Extraction

class FeatureExtractor:
    """Extract features using pre-trained CNN"""

    def __init__(self, model_name='resnet50'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self):
        """Load pre-trained model"""
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """Get image transformation"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image):
        """Extract features from single image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)

        features = features.squeeze().cpu().numpy()
        return features / (np.linalg.norm(features) + 1e-8)

    def extract_dataset_features(self, df, image_loader):
        """Extract features for entire dataset"""
        features = []

        for idx in tqdm(range(len(df)), desc="Extracting features"):
            img_path = df.iloc[idx]['image_path']
            img = image_loader.load_image(img_path)

            if img is not None:
                feat = self.extract_features(img)
                features.append(feat)
            else:
                features.append(np.zeros(self.model.fc.in_features if hasattr(self.model, 'fc') else 2048))

        return np.vstack(features)

# Extract features (or load if already computed)
if df is not None:
    feature_extractor = FeatureExtractor()
    features_path = config.FEATURES_DIR / "notebook_features.pkl"

    if features_path.exists():
        print("Loading pre-computed features...")
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features shape: {features.shape}")
    else:
        print("Extracting features (this may take a while)...")
        features = feature_extractor.extract_dataset_features(df, data_loader)

        # Save features
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Features saved. Shape: {features.shape}")

# ## 6. Similarity Search

class SimilaritySearch:
    """Handle similarity search and recommendations"""

    def __init__(self, features):
        self.features = normalize(features, norm='l2')

    def find_similar(self, query_idx, k=10):
        """Find k most similar items"""
        query_feat = self.features[query_idx].reshape(1, -1)
        similarities = cosine_similarity(query_feat, self.features)[0]

        # Get top-k indices (excluding query itself)
        top_indices = np.argsort(similarities)[::-1][1:k+1]
        top_scores = similarities[top_indices]

        return top_indices, top_scores

    def find_diverse(self, query_idx, k=10, diversity=0.3):
        """Find diverse recommendations using MMR"""
        candidates_idx, _ = self.find_similar(query_idx, k=k*3)

        selected = []
        remaining = list(candidates_idx)

        # Select first (most similar)
        selected.append(remaining[0])
        remaining.remove(remaining[0])

        # Iteratively select diverse items
        while len(selected) < k and remaining:
            query_sim = cosine_similarity(
                self.features[query_idx].reshape(1, -1),
                self.features[remaining]
            )[0]

            selected_feats = self.features[selected]
            max_sim_to_selected = cosine_similarity(
                self.features[remaining],
                selected_feats
            ).max(axis=1)

            mmr_scores = (1 - diversity) * query_sim - diversity * max_sim_to_selected
            best_idx = remaining[np.argmax(mmr_scores)]

            selected.append(best_idx)
            remaining.remove(best_idx)

        return np.array(selected)

if df is not None and 'features' in locals():
    search_engine = SimilaritySearch(features)

# ## 7. Visualization Functions

def display_products(indices, df, title="Products", n_cols=4):
    """Display products in grid"""
    n_items = len(indices)
    n_rows = (n_items + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3.5))
    axes = axes.flatten() if n_items > 1 else [axes]

    for i, idx in enumerate(indices):
        product = df.iloc[idx]
        img_path = product['image_path']

        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)

        axes[i].set_title(f"{product['productDisplayName'][:30]}...", fontsize=10)
        axes[i].text(0.5, -0.15, f"Category: {product['masterCategory']}",
                    transform=axes[i].transAxes, ha='center', fontsize=8)
        axes[i].text(0.5, -0.25, f"Color: {product['baseColour']}",
                    transform=axes[i].transAxes, ha='center', fontsize=8)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_items, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ## 8. Interactive Recommendation System

if df is not None and 'search_engine' in locals():
    def show_recommendations(product_name):
        """Interactive recommendation function"""
        # Find product
        matches = df[df['productDisplayName'].str.contains(product_name, case=False)]

        if len(matches) == 0:
            print(f"No product found matching '{product_name}'")
            return

        query_idx = matches.index[0]
        query_product = df.iloc[query_idx]

        # Display query product
        print(f"Query Product: {query_product['productDisplayName']}")
        display_products([query_idx], df, "Query Product", n_cols=1)

        # Find similar items
        similar_idx, scores = search_engine.find_similar(query_idx, k=8)
        display_products(similar_idx, df, "Similar Items", n_cols=4)

        # Find diverse recommendations
        diverse_idx = search_engine.find_diverse(query_idx, k=8, diversity=0.4)
        display_products(diverse_idx, df, "Diverse Recommendations", n_cols=4)

    # Create interactive widget
    product_names = df['productDisplayName'].head(100).tolist()

    dropdown = widgets.Dropdown(
        options=product_names,
        value=product_names[0],
        description='Product:',
        layout=widgets.Layout(width='500px')
    )

    button = widgets.Button(description="Get Recommendations")
    output = widgets.Output()

    def on_button_click(b):
        with output:
            clear_output()
            show_recommendations(dropdown.value)

    button.on_click(on_button_click)

    display(dropdown, button, output)

# ## 9. Summary Statistics

if df is not None:
    print("\n=== Dataset Summary ===")
    print(f"Total Products: {len(df)}")
    print(f"Categories: {df['masterCategory'].nunique()}")
    print(f"Colors: {df['baseColour'].nunique()}")
    print(f"Genders: {df['gender'].nunique()}")
    print(f"Seasons: {df['season'].nunique()}")

    if 'features' in locals():
        print(f"\nFeature Dimensions: {features.shape}")
        print(f"Feature Extraction Model: ResNet50")

print("\n✅ Notebook execution complete!")
print("You can now use the interactive widgets above to explore recommendations.")
