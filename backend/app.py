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

# Initialize face recognition system
logger.info("Initializing Face Recognition System...")
face_recognition_system = FaceRecognitionSystem()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database_size': len(face_recognition_system.id_embeddings),
        'threshold': face_recognition_system.threshold,
        'registered_faces': list(face_recognition_system.id_embeddings.keys())
    })

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

        # Process first image
        face1 = face_recognition_system.extract_face(file1)
        emb1 = face_recognition_system.get_embedding(face1)

        # Process second image
        face2 = face_recognition_system.extract_face(file2)
        emb2 = face_recognition_system.get_embedding(face2)

        # Compare embeddings
        is_match, distance = face_recognition_system.compare_embeddings(emb1, emb2)
        
        # The distance is a measure of dissimilarity, so similarity is 1 - distance.
        # The threshold is on the distance, so a smaller distance is better.
        # Let's convert distance to a more intuitive similarity percentage.
        # A simple linear conversion could be (threshold - distance) / threshold.
        # When distance is 0 (perfect match), similarity is 100%.
        # When distance is at the threshold, similarity is 0%.
        # Anything beyond the threshold is a non-match.
        
        similarity_score = max(0, (face_recognition_system.threshold - distance) / face_recognition_system.threshold) * 100

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
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        try:
            face = face_recognition_system.extract_face(file)
            embedding = face_recognition_system.get_embedding(face)

            # Find the best match in the database
            best_match_name = "Unknown"
            min_distance = float('inf')

            for name, id_emb in face_recognition_system.id_embeddings.items():
                is_match, distance = face_recognition_system.compare_embeddings(embedding, id_emb)
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
            
    except Exception as e:
        logger.error(f"Recognition endpoint error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/database', methods=['GET'])
def get_database_info():
    """Get information about the ID database"""
    return jsonify({
        'total_ids': len(face_recognition_system.id_embeddings),
        'id_names': list(face_recognition_system.id_embeddings.keys()),
        'threshold': face_recognition_system.threshold
    })

@app.route('/reload_database', methods=['POST'])
def reload_database():
    """Reload the ID database"""
    try:
        global face_recognition_system
        face_recognition_system = FaceRecognitionSystem()
        return jsonify({
            'success': True,
            'message': f'Database reloaded with {len(face_recognition_system.id_embeddings)} IDs'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Face Recognition API Server...")
    logger.info(f"Database loaded with {len(face_recognition_system.id_embeddings)} IDs")
    
    if len(face_recognition_system.id_embeddings) == 0:
        logger.warning("No ID embeddings found! Please add ID photos to database/ids/")
    
    app.run(host='0.0.0.0', port=5000, debug=True)