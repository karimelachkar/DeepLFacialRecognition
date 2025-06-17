# Face Recognition App

This project is a one-shot face recognition system designed for a mobile-first experience. It consists of a Python/Flask backend deployed on Google Cloud and a React Native frontend for iOS and Android.

## 🚀 Quick Start: Running the Mobile App (for Presentation)

These instructions will guide you through running the mobile application on your phone for a live demonstration.

**Prerequisites:**

- An iOS or Android phone
- Wi-Fi connection (your phone and computer must be on the same network)

**Step 1: Install the Expo Go App**

- On your phone, download the **Expo Go** app from the Apple App Store or Google Play Store.

**Step 2: Navigate to the Frontend Directory**

- Open a terminal on your computer and navigate to the project's `frontend` directory:
  ```bash
  cd path/to/your/project/frontend
  ```

**Step 3: Install Dependencies**

- If you haven't done so already, install the necessary packages.
  ```bash
  npm install
  ```

**Step 4: Start the Development Server**

- Run the following command to start the local server. This will generate a unique QR code in your terminal.
  ```bash
  npm start
  ```

**Step 5: Launch the App**

- Open the **Expo Go** app on your phone and scan the QR code from your terminal. The app will automatically build and load onto your phone.

You can now use the app to register a new ID and perform face recognition against the live backend service.

---

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
```
