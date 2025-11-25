"""
Utility functions for image processing
"""
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# Constants matching your training
IMG_SIZE = 224
NUM_CLASSES = 120


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for ResNet50 model - MUST MATCH TRAINING!
    
    Args:
        image_bytes: Raw image bytes from upload
        
    Returns:
        Preprocessed image array ready for prediction
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (in case of RGBA or grayscale)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 (ResNet50 input size)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # CRITICAL: Use ResNet50 preprocessing (same as training!)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(image_array)
    
    return preprocessed


def format_breed_name(breed: str) -> str:
    """
    Format breed name for display
    
    Args:
        breed: Raw breed name (e.g., 'golden_retriever')
        
    Returns:
        Formatted breed name (e.g., 'Golden Retriever')
    """
    # Replace underscores with spaces and title case
    return breed.replace('_', ' ').title()
