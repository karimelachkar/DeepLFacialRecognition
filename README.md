# Face Recognition App

A one-shot face recognition system using FaceNet that compares live photos against a database of registered ID photos.

## Features

- Face detection and extraction using MTCNN
- Face recognition using pre-trained FaceNet model
- REST API for mobile app integration
- Confidence scoring and similarity matching

## Setup

### Requirements

- Python 3.11+ (recommended)
- Virtual environment (recommended)

### Installation

```bash
# Clone and setup
git clone <your-repo>
cd face-recognition-app/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Add ID Photos

```bash
# Add your ID photos to the database
mkdir -p database/ids
# Copy your ID photos (JPG/PNG) to database/ids/
# Example: database/ids/john_doe.jpg
```

### Run

```bash
python app.py
```

Server starts at `http://localhost:5000`

## API Endpoints

- `GET /health` - Check server status
- `POST /recognize` - Upload image for face recognition
- `GET /database` - View registered faces

## Testing

```bash
# Check server health
curl http://localhost:5000/health

# Test recognition (replace with your image path)
curl -X POST -F "image=@path/to/test/photo.jpg" http://localhost:5000/recognize
```

## Project Structure

```
backend/
├── app.py                 # Flask API server
├── face_recognition.py    # Face recognition logic
├── requirements.txt       # Dependencies
├── database/
│   ├── ids/              # ID photos (add your photos here)
│   └── embeddings.json   # Pre-computed embeddings (auto-generated)
└── uploads/              # Temporary uploads
