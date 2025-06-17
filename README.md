# Face Recognition App

This project is a one-shot face recognition system designed for a mobile-first experience. It consists of a Python/Flask backend deployed on Google Cloud and a React Native frontend for iOS and Android.

## 🚀 Quick Start (for Presentation)

These instructions explain how to install the app directly onto an Android phone for a demonstration.

**Step 1: Open the Installation Link**

- On an Android phone (or emulator), open the following link in a web browser:
- [https://expo.dev/accounts/karimelachkar/projects/frontend/builds/c4b70092-f17d-4311-a300-49747a1d48df](https://expo.dev/accounts/karimelachkar/projects/frontend/builds/c4b70092-f17d-4311-a300-49747a1d48df)

**Step 2: Download and Install**

- The link will open a page with a "Download" button. Tap it to download the app's installer file (`.apk`).
- Once downloaded, open the file and follow the on-screen prompts to install the app. You may need to grant your browser permission to install apps from unknown sources.

**Step 3: Use the App**

- Open the "frontend" app from your phone's app drawer.
- The app will ask you to provide two images:
  1.  **ID Picture:** The reference photo.
  2.  **Verification Picture:** The photo to compare against the ID.
- The result of the comparison will be displayed on the screen.

---

## 💻 For Developers: Running Locally

If you want to run the app in a local development environment, follow these steps.

**Prerequisites:**

- An iOS or Android phone with the **Expo Go** app installed.
- Your computer and phone must be on the same Wi-Fi network.

**Step 1: Navigate to the Frontend Directory**

- Open a terminal on your computer and navigate to the project's `frontend` directory:
  ```bash
  cd path/to/your/project/frontend
  ```

**Step 2: Install Dependencies**

- If you haven't done so already, install the necessary packages.
  ```bash
  npm install
  ```

**Step 3: Start the Development Server**

- Run the following command. This will generate a unique QR code in your terminal.
  ```bash
  npm start
  ```

**Step 4: Launch the App**

- Open the **Expo Go** app on your phone and scan the QR code from your terminal. The app will automatically build and load.

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
