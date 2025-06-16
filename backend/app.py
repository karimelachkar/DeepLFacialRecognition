import os
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from face_recognition import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('database/ids', exist_ok=True)

# LAZY LOADING: Defer initialization of the face recognition system
_face_recognition_system = None

def get_face_recognition_system():
    """Initializes and returns a singleton instance of the FaceRecognitionSystem."""
    global _face_recognition_system
    if _face_recognition_system is None:
        logger.info("Initializing Face Recognition System for the first time...")
        _face_recognition_system = FaceRecognitionSystem()
        logger.info("Face Recognition System initialized.")
    return _face_recognition_system

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    system = get_face_recognition_system()
    return jsonify({
        'status': 'healthy',
        'database_size': len(system.id_embeddings),
        'threshold': system.threshold,
        'registered_faces': list(system.id_embeddings.keys())
    })

@app.route('/register', methods=['POST'])
def register_face():
    """Endpoint to register a new face."""
    try:
        if 'image' not in request.files or 'name' not in request.form:
            return jsonify({'error': 'Image file and name are required'}), 400

        file = request.files['image']
        name = request.form['name']

        if file.filename == '' or name == '':
            return jsonify({'error': 'Image and name cannot be empty'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Extract face and create embedding
        system = get_face_recognition_system()
        face = system.extract_face(file)
        if face is None:
            return jsonify({'error': 'No face could be detected in the image.'}), 400
            
        embedding = system.get_embedding(face)

        # Add to in-memory database
        system.id_embeddings[name] = embedding
        logger.info(f"Registered new face: {name}")

        # Save the updated database to file
        system.save_database()

        return jsonify({'success': True, 'name': name, 'message': 'Face registered successfully.'})

    except ValueError as ve:
        logger.error(f"Registration error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Registration endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_face():
    """One-to-one face verification endpoint"""
    try:
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': 'Two image files are required'}), 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': 'No file selected for one or both images'}), 400

        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400

        system = get_face_recognition_system()
        # Process first image
        face1 = system.extract_face(file1)
        emb1 = system.get_embedding(face1)

        # Process second image
        face2 = system.extract_face(file2)
        emb2 = system.get_embedding(face2)

        # Compare embeddings
        is_match, distance = system.compare_embeddings(emb1, emb2)
        
        # The distance is a measure of dissimilarity, so similarity is 1 - distance.
        # The threshold is on the distance, so a smaller distance is better.
        # Let's convert distance to a more intuitive similarity percentage.
        # A simple linear conversion could be (threshold - distance) / threshold.
        # When distance is 0 (perfect match), similarity is 100%.
        # When distance is at the threshold, similarity is 0%.
        # Anything beyond the threshold is a non-match.
        
        similarity_score = max(0, (system.threshold - distance) / system.threshold) * 100

        return jsonify({
            'verified': bool(is_match),
            'distance': float(distance),
            'similarity': float(similarity_score)
        })

    except ValueError as ve:
        logger.error(f"Verification error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Verification endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Main face recognition endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        system = get_face_recognition_system()
        face = system.extract_face(file)
        if face is None:
            return jsonify({'error': 'No face could be detected in the image.'}), 400

        embedding = system.get_embedding(face)

        if not system.id_embeddings:
            return jsonify({'name': 'Unknown', 'distance': None, 'error': 'No IDs have been registered in the database.'})

        # Find the best match in the database
        best_match_name = "Unknown"
        min_distance = float('inf')

        for name, id_emb in system.id_embeddings.items():
            is_match, distance = system.compare_embeddings(embedding, id_emb)
            if is_match and distance < min_distance:
                min_distance = distance
                best_match_name = name

        return jsonify({
            'name': best_match_name,
            'distance': float(min_distance) if min_distance != float('inf') else None
        })

    except ValueError as ve:
        logger.error(f"Recognition error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Recognition endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/database', methods=['GET'])
def get_database_info():
    """Get information about the ID database"""
    system = get_face_recognition_system()
    return jsonify({
        'total_ids': len(system.id_embeddings),
        'id_names': list(system.id_embeddings.keys()),
        'threshold': system.threshold
    })

@app.route('/reload_database', methods=['POST'])
def reload_database():
    """Reload the ID database"""
    try:
        global _face_recognition_system
        _face_recognition_system = FaceRecognitionSystem()
        return jsonify({
            'success': True,
            'message': f'Database reloaded with {len(_face_recognition_system.id_embeddings)} IDs'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Face Recognition API Server in debug mode...")
    # In debug mode, we don't lazy load so that we can see initialization errors immediately.
    get_face_recognition_system()
    app.run(host='0.0.0.0', port=5000, debug=True)