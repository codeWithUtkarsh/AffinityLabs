"""
Configuration file for Fashion Shopping Assistant
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, FEATURES_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset paths
IMAGES_DIR = DATA_DIR / "images"
STYLES_CSV = DATA_DIR / "styles.csv"

# Feature extraction settings
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
FEATURE_DIM = 2048  # ResNet50 feature dimension
MODEL_NAME = "resnet50"

# Search settings
TOP_K = 10  # Number of similar items to retrieve
MIN_SIMILARITY = 0.5  # Minimum similarity threshold

# UI settings
ITEMS_PER_ROW = 4
MAX_DISPLAY_ITEMS = 20

# Categories for filtering
MASTER_CATEGORIES = [
    "All", "Apparel", "Accessories", "Footwear", "Personal Care",
    "Free Items", "Sporting Goods", "Home"
]

GENDER_OPTIONS = ["All", "Men", "Women", "Boys", "Girls", "Unisex"]

COLOR_OPTIONS = [
    "All", "Black", "Blue", "White", "Brown", "Grey", "Red", "Green",
    "Yellow", "Pink", "Orange", "Purple", "Navy Blue", "Beige"
]

SEASON_OPTIONS = ["All", "Summer", "Winter", "Spring", "Fall"]

# Kaggle dataset info
KAGGLE_DATASET = "paramaggarwal/fashion-product-images-small"
