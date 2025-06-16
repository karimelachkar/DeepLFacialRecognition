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
        
        # Save uploaded file temporarily
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Perform face recognition using your exact Colab logic
            result = face_recognition_system.recognize_face(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
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