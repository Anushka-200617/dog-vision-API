"""
Model loading and prediction logic
"""
import tensorflow as tf
import numpy as np
from typing import List, Tuple
from pathlib import Path

from app.utils import preprocess_image, format_breed_name, IMG_SIZE, NUM_CLASSES


class DogBreedPredictor:
    """Handler for dog breed prediction using transfer learning model"""
    
    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the predictor with model and labels
        
        Args:
            model_path: Path to the saved model weights
            labels_path: Path to the breed labels file
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = []
        
        # Load model and labels
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load the trained Keras model - Build architecture and load weights"""
        try:
            print(f"📦 Building model architecture...")
            
            # Build the exact architecture from training
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
            from tensorflow.keras import Sequential
            
            # Create Sequential model (matches your training)
            self.model = Sequential([
                ResNet50(
                    include_top=False,
                    weights=None,  # We'll load your trained weights
                    input_shape=(224, 224, 3)
                ),
                GlobalAveragePooling2D(),
                Dense(120, activation='softmax')
            ])
            
            # Freeze ResNet50 layers (match training)
            self.model.layers[0].trainable = False
            
            print("✅ Architecture built")
            print(f"   Total layers: {len(self.model.layers)}")
            
            # Load your trained weights
            print(f"📥 Loading weights from: {self.model_path}")
            self.model.load_weights(self.model_path)
            print("✅ Weights loaded!")
            
            # Compile (needed for predictions)
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Model ready!")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            print(f"   TensorFlow version: {tf.__version__}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to load model: {e}")
    
    def _load_labels(self):
        """Load breed labels from file"""
        try:
            print(f"📋 Loading labels from: {self.labels_path}")
            
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                # Read lines and strip whitespace
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(self.labels) != NUM_CLASSES:
                print(f"⚠️  Warning: Expected {NUM_CLASSES} labels, got {len(self.labels)}")
            
            print(f"✅ Loaded {len(self.labels)} breed labels")
            print(f"   First breed: {self.labels[0]}")
            print(f"   Last breed: {self.labels[-1]}")
            
        except Exception as e:
            print(f"❌ Error loading labels: {e}")
            raise Exception(f"Failed to load labels: {e}")
    
    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict dog breed from image
        
        Args:
            image_bytes: Raw image bytes
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (breed_name, confidence_score)
        """
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction (verbose=0 to suppress output)
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top K predictions
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        # Create results with breed names and confidence scores
        results = [
            (self.labels[idx], float(predictions[0][idx]))
            for idx in top_indices
        ]
        
        return results
    
    def get_all_breeds(self) -> List[str]:
        """Get all breed labels"""
        return self.labels.copy()


# Global predictor instance (loaded once at startup)
_predictor: DogBreedPredictor = None


def get_predictor() -> DogBreedPredictor:
    """
    Get or create predictor instance (Singleton pattern)
    
    Returns:
        DogBreedPredictor instance
    """
    global _predictor
    
    if _predictor is None:
        # Get paths relative to project root
        base_dir = Path(__file__).parent.parent
        
        # Use your trained model weights
        # Change this to match your weights file name
        model_path = base_dir / "model" / "dog_breed_WEIGHTS_ONLY.weights.h5"
        labels_path = base_dir / "model" / "labels.txt"
        
        # Validate files exist
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights file not found: {model_path}\n"
                f"Please run the Colab script to save weights and download the file."
            )
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        # Create predictor
        _predictor = DogBreedPredictor(
            model_path=str(model_path),
            labels_path=str(labels_path)
        )
    
    return _predictor
