# Face Recognition Web App 🔍

A real-time facial recognition system using webcam-based registration and login functionality. Built using Python (Flask, OpenCV) for the backend and a fully responsive HTML + Tailwind CSS frontend.

## 🌐 Live Demo

- **Frontend**: [https://primordial579.github.io/Facerecog/facerecog.html](https://primordial579.github.io/Face-recognition/facerecog.html)


## ✨ Features

- 📸 Register your face via webcam with 50 auto-captured samples
- 🔐 Login using face recognition with high accuracy
- 🧠 PCA for feature compression
- 🧪 KNN classifier for prediction
- 🚀 Deployed on Render (backend) and GitHub Pages (frontend)
- 🔄 Responsive and animated UI (TailwindCSS)

## 🛠️ Tech Stack

### 🧠 Backend
- Python + Flask
- Flask-CORS
- OpenCV
- scikit-learn (PCA, KNN)
- Pillow (for base64 to image conversion)

### 🌐 Frontend
- HTML + CSS (Tailwind)
- Vanilla JavaScript
- Webcam integration with canvas capture

## 📦 Installation (Backend)

1. Clone the repository
2. Navigate to the backend folder (if separated)
3. Install dependencies:
   ```bash
   pip install flask flask-cors opencv-python-headless scikit-learn numpy pillow

python facerecog.py

🧪 API Endpoints
Method	Endpoint	Description
POST	/register	Register new user with name + images
POST	/login	Identify user by webcam image
GET	/health	Health check (status, model)
