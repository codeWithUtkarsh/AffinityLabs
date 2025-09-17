"""
Feature extraction module using pre-trained CNN models
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from pathlib import Path
import config

class FeatureExtractor:
    """
    Extract features from images using pre-trained models
    """

    def __init__(self, model_name='resnet50', device=None):
        """
        Initialize feature extractor

        Args:
            model_name: Name of pre-trained model to use
            device: Device to run model on
        """
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _load_model(self):
        """
        Load pre-trained model
        """
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            model = model.features
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        model = model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """
        Get image transformation pipeline
        """
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
        """
        Extract features from a single image
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)

        # Flatten features
        features = features.squeeze().cpu().numpy()

        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-8)

        return features

    def extract_batch_features(self, images):
        """
        Extract features from a batch of images
        """
        batch_tensors = []

        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            img_tensor = self.transform(img)
            batch_tensors.append(img_tensor)

        # Stack into batch
        batch = torch.stack(batch_tensors).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(batch)

        # Process features
        features = features.squeeze().cpu().numpy()

        # Handle single image case
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Normalize each feature vector
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)

        return features

    def extract_dataset_features(self, data_loader, save_path=None):
        """
        Extract features for entire dataset
        """
        all_features = []
        all_indices = []

        print(f"Extracting features using {self.model_name}...")

        # Process in batches
        batch_size = config.BATCH_SIZE
        n_batches = (len(data_loader.df) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data_loader.df))
            indices = list(range(start_idx, end_idx))

            # Load batch of images
            images, valid_indices = data_loader.get_batch_images(indices)

            if images:
                # Extract features
                features = self.extract_batch_features(images)
                all_features.append(features)
                all_indices.extend(valid_indices)

        # Concatenate all features
        if all_features:
            all_features = np.vstack(all_features)
        else:
            all_features = np.array([])

        # Save features if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'wb') as f:
                pickle.dump({
                    'features': all_features,
                    'indices': all_indices,
                    'model_name': self.model_name
                }, f)

            print(f"Features saved to {save_path}")

        return all_features, all_indices

    def load_features(self, features_path):
        """
        Load pre-computed features
        """
        with open(features_path, 'rb') as f:
            data = pickle.load(f)

        return data['features'], data['indices']

    def compute_similarity(self, features1, features2):
        """
        Compute cosine similarity between two feature vectors
        """
        # Ensure features are normalized
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)

        # Compute cosine similarity
        similarity = np.dot(features1, features2)

        return similarity
