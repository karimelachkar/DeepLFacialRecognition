import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.threshold = 0.97
        
        # Load FaceNet model
        self.logger.info("Loading FaceNet model...")
        from keras_facenet import FaceNet
        self.embedder = FaceNet()
        self.logger.info("FaceNet model loaded.")
        
        # Initialize embeddings dictionary
        self.id_embeddings = {}
        
        # Load embeddings from JSON file
        self.load_database()

    
    def extract_face(self, filename, required_size=(160, 160)):
        """
        Extract face from image - EXACT copy from your Colab notebook
        """
        if isinstance(filename, str):
            image = Image.open(filename)
        else:
            # Handle file-like object
            image = Image.open(filename)

        # Check and apply rotation based on EXIF data.
        # This block of code avoids the issue of rotation of images when being imported
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)

        except (AttributeError, KeyError, IndexError):
            # Cases: image doesn't have getexif
            pass # No EXIF data or orientation tag

        image = image.convert('RGB')
        pixels = np.asarray(image)
        results = self.detector.detect_faces(pixels)

        if not results:
            raise ValueError(f"No face detected in the image")

        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = pixels[y1:y2, x1:x2]
        face = cv2.resize(face, required_size)
        return face
    
    def get_embedding(self, face_array):
        """
        Generate embedding - EXACT copy from your Colab
        """
        # Wrap in a list since .embeddings() expects a batch
        embedding = self.embedder.embeddings([face_array])[0]
        return embedding
    
    def ensure_normalized(self, v, tolerance=1e-3):
        """
        Normalize vector - EXACT copy from your Colab
        """
        length = norm(v)
        # Check if already normalized (L2 norm ≈ 1)
        if abs(length - 1.0) < tolerance:
            return v
        else:
            return v / length
    
    def compare_embeddings(self, embedding1, embedding2, threshold=None):
        """
        Compare embeddings - EXACT copy from your Colab
        """
        if threshold is None:
            threshold = self.threshold
            
        # Normalize embeddings if needed
        emb1 = self.ensure_normalized(embedding1)
        emb2 = self.ensure_normalized(embedding2)

        # Compute distance
        distance = norm(emb1 - emb2)

        # Determine result
        is_match = distance < threshold

        return is_match, distance
    
    def load_database(self):
        """Load ID embeddings from embeddings.json file"""
        
        embeddings_file = "database/embeddings.json"
        
        try:
            self.logger.info(f"Loading embeddings from {embeddings_file}...")
            
            # Check if file exists
            if not os.path.exists(embeddings_file):
                self.logger.warning(f"Embeddings file not found: {embeddings_file}")
                self.logger.warning("Run create_embeddings.py to generate embeddings from ID photos")
                return
            
            # Load JSON data
            with open(embeddings_file, 'r') as f:
                data = json.load(f)
            
            # Extract embeddings
            if "embeddings" in data:
                embeddings_data = data["embeddings"]
                
                # Convert each embedding back to numpy array
                for person_name, person_data in embeddings_data.items():
                    embedding_list = person_data["embedding"]
                    embedding_array = np.array(embedding_list)
                    
                    # Store in id_embeddings dictionary
                    self.id_embeddings[person_name] = embedding_array
                    
                    self.logger.debug(f"Loaded embedding for {person_name}: {embedding_array.shape}")
                
                self.logger.info(f"✅ Successfully loaded {len(self.id_embeddings)} ID embeddings")
                self.logger.info(f"Loaded IDs: {list(self.id_embeddings.keys())}")
                
                # Log metadata if available
                if "metadata" in data:
                    metadata = data["metadata"]
                    self.logger.info(f"Embedding dimension: {metadata.get('embedding_dimension')}")
                    self.logger.info(f"Model used: {metadata.get('model')}")
            
            else:
                self.logger.error("Invalid embeddings file format - missing 'embeddings' key")
                
        except FileNotFoundError:
            self.logger.error(f"Embeddings file not found: {embeddings_file}")
            self.logger.error("Please run create_embeddings.py first to generate embeddings")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON file: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error loading embeddings: {e}")
            import traceback
            self.logger.error(traceback.format_exc())