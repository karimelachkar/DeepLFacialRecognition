import os
import json
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_face(filename, required_size=(160, 160)):
    """Extract face from image (from your Colab code)"""
    
    from PIL import Image, ExifTags
    import numpy as np
    import cv2
    from mtcnn.mtcnn import MTCNN
    
    logger.info(f"Processing: {filename}")
    
    image = Image.open(filename)
    
    # Check and apply rotation based on EXIF data
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = dict(image._getexif().items())
            
            if orientation in exif:
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
                    
    except (AttributeError, KeyError, IndexError):
        pass  # No EXIF data or orientation tag
    
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    
    if not results:
        logger.warning(f"No face detected in {filename}")
        raise ValueError(f"No face detected in the image: {filename}")
    
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    face = pixels[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    
    logger.info(f"Face extracted successfully from {filename}")
    return face

def ensure_normalized(v, tolerance=1e-3):
    """Normalize vector (from your Colab code)"""
    from numpy.linalg import norm
    
    length = norm(v)
    if abs(length - 1.0) < tolerance:
        return v
    else:
        return v / length

def create_embeddings_from_ids():
    """Create embeddings.json from all ID photos"""
    
    logger.info("="*60)
    logger.info("CREATING EMBEDDINGS.JSON FROM ID PHOTOS")
    logger.info("="*60)
    
    # Paths
    ids_folder = "database/ids"
    embeddings_file = "database/embeddings.json"
    
    # Check if IDs folder exists
    if not os.path.exists(ids_folder):
        logger.warning(f"IDs folder not found: {ids_folder}. Creating it.")
        os.makedirs(ids_folder)
    
    # Create database folder if it doesn't exist
    os.makedirs("database", exist_ok=True)
    
    # Get all image files
    image_files = []
    for filename in os.listdir(ids_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(filename)
    
    logger.info(f"Found {len(image_files)} ID photos to process")
    
    if not image_files:
        logger.warning("No image files found in the IDs folder. Creating an empty embeddings file.")
        # Create an empty embeddings.json
        final_data = {
            "metadata": {
                "total_ids": 0,
                "embedding_dimension": None,
                "model": "FaceNet",
                "normalization": "L2"
            },
            "embeddings": {}
        }
        with open(embeddings_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        logger.info(f"✅ Empty embeddings file created at {embeddings_file}")
        return True
    
    logger.info(f"ID files: {image_files}")
    
    # Load FaceNet model
    logger.info("Loading FaceNet model...")
    from keras_facenet import FaceNet
    embedder = FaceNet()
    logger.info("FaceNet model loaded successfully")
    
    # Process each ID photo
    embeddings_data = {}
    successful_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"\nProcessing {i}/{len(image_files)}: {image_file}")
        
        try:
            # Extract person name from filename
            person_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(ids_folder, image_file)
            
            logger.info(f"  Person: {person_name}")
            
            # Extract face
            face = extract_face(image_path)
            logger.info(f"  Face extracted: {face.shape}")
            
            # Generate embedding
            embedding = embedder.embeddings([face])[0]
            logger.info(f"  Embedding generated: {embedding.shape}")
            
            # Normalize embedding
            embedding = ensure_normalized(embedding)
            logger.info(f"  Embedding normalized")
            
            # Convert to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Store in data dictionary
            embeddings_data[person_name] = {
                "embedding": embedding_list,
                "source_file": image_file,
                "embedding_shape": list(embedding.shape)
            }
            
            successful_count += 1
            logger.info(f"  ✅ Successfully processed {person_name}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to process {image_file}: {e}")
            # Continue with other images
            continue
    
    # Save embeddings to JSON file
    if embeddings_data:
        logger.info(f"\nSaving embeddings to {embeddings_file}...")
        
        # Add metadata
        final_data = {
            "metadata": {
                "total_ids": len(embeddings_data),
                "embedding_dimension": list(embeddings_data[list(embeddings_data.keys())[0]]["embedding_shape"]),
                "created_from": ids_folder,
                "model": "FaceNet",
                "normalization": "L2"
            },
            "embeddings": embeddings_data
        }
        
        with open(embeddings_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"✅ Embeddings saved successfully!")
        logger.info(f"📊 Summary:")
        logger.info(f"   Total IDs processed: {successful_count}/{len(image_files)}")
        logger.info(f"   Embeddings file: {embeddings_file}")
        logger.info(f"   File size: {os.path.getsize(embeddings_file)} bytes")
        logger.info(f"   IDs included: {list(embeddings_data.keys())}")
        
        return True
    else:
        logger.error("❌ No embeddings generated!")
        return False

def verify_embeddings_file():
    """Verify that the embeddings file was created correctly"""
    
    logger.info("\n" + "="*60)
    logger.info("VERIFYING EMBEDDINGS.JSON FILE")
    logger.info("="*60)
    
    embeddings_file = "database/embeddings.json"
    
    if not os.path.exists(embeddings_file):
        logger.error(f"❌ Embeddings file not found: {embeddings_file}")
        return False
    
    try:
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"✅ Embeddings file loaded successfully")
        
        # Check structure
        if "metadata" in data and "embeddings" in data:
            metadata = data["metadata"]
            embeddings = data["embeddings"]
            
            logger.info(f"📊 Metadata:")
            logger.info(f"   Total IDs: {metadata.get('total_ids')}")
            logger.info(f"   Embedding dimension: {metadata.get('embedding_dimension')}")
            logger.info(f"   Model: {metadata.get('model')}")
            
            logger.info(f"📊 Embeddings:")
            logger.info(f"   Number of people: {len(embeddings)}")
            
            for name, data in embeddings.items():
                embedding_length = len(data["embedding"])
                source_file = data["source_file"]
                logger.info(f"   {name}: {embedding_length}D vector from {source_file}")
            
            logger.info(f"✅ Embeddings file structure is correct!")
            return True
        else:
            logger.error(f"❌ Invalid file structure")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error reading embeddings file: {e}")
        return False

def test_flask_app_loading():
    """Test if your Flask app can now load the embeddings"""
    
    logger.info("\n" + "="*60)
    logger.info("TESTING FLASK APP LOADING")
    logger.info("="*60)
    
    try:
        # Try to simulate what your Flask app does
        embeddings_file = "database/embeddings.json"
        
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        
        embeddings = data["embeddings"]
        
        # Convert back to numpy arrays (like your Flask app should do)
        loaded_embeddings = {}
        for name, person_data in embeddings.items():
            embedding_array = np.array(person_data["embedding"])
            loaded_embeddings[name] = embedding_array
        
        logger.info(f"✅ Successfully simulated Flask app loading")
        logger.info(f"📊 Loaded {len(loaded_embeddings)} ID embeddings")
        logger.info(f"🆔 IDs: {list(loaded_embeddings.keys())}")
        
        # Test embedding properties
        first_embedding = list(loaded_embeddings.values())[0]
        logger.info(f"📏 Embedding shape: {first_embedding.shape}")
        logger.info(f"📐 Embedding norm: {np.linalg.norm(first_embedding):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error simulating Flask app loading: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting embeddings creation process...\n")
    
    # Step 1: Create embeddings from ID photos
    success = create_embeddings_from_ids()
    
    if success:
        # Step 2: Verify the file was created correctly
        verify_embeddings_file()
        
        # Step 3: Test Flask app loading simulation
        test_flask_app_loading()
        
        print(f"\n🎉 SUCCESS! Your embeddings.json file is ready!")
        print(f"📁 Location: database/embeddings.json")
        print(f"🔄 Now restart your Flask app - it should load the IDs successfully!")
        
    else:
        print(f"\n❌ Failed to create embeddings.json")
        print(f"Check the error messages above and fix any issues with your ID photos")