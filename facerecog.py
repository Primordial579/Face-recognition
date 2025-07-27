from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import base64
import io
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model_file = 'face_model.pkl'
        self.pca_file = 'pca_model.pkl'
        self.labels_file = 'labels.pkl'
        self.min_samples = 10
        self.confidence_threshold = 0.7
        
        # Load existing models if they exist
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.pca_file) and os.path.exists(self.labels_file):
                with open(self.model_file, 'rb') as f:
                    self.knn_model = pickle.load(f)
                with open(self.pca_file, 'rb') as f:
                    self.pca = pickle.load(f)
                with open(self.labels_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Models loaded successfully")
            else:
                self.knn_model = None
                self.pca = None
                self.label_encoder = {}
                logger.info("No existing models found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.knn_model = None
            self.pca = None
            self.label_encoder = {}
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.knn_model, f)
            with open(self.pca_file, 'wb') as f:
                pickle.dump(self.pca, f)
            with open(self.labels_file, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def detect_faces(self, image):
        """Detect faces in an image and return face regions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces, gray
    
    def extract_face_features(self, gray_image, face_coords):
        """Extract face region and resize to standard size"""
        x, y, w, h = face_coords
        face_region = gray_image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (100, 100))
        return face_resized.flatten()
    
    def base64_to_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return cv_image
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            return None
    
    def add_new_face(self, name, face_images):
        """Add a new person's face data to the system"""
        try:
            features = []
            valid_samples = 0
            
            for img_data in face_images:
                image = self.base64_to_image(img_data)
                if image is None:
                    continue
                
                faces, gray = self.detect_faces(image)
                
                if len(faces) > 0:
                    # Use the largest face detected
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    face_features = self.extract_face_features(gray, largest_face)
                    features.append(face_features)
                    valid_samples += 1
            
            if valid_samples < self.min_samples:
                return False, f"Not enough valid face samples. Got {valid_samples}, need at least {self.min_samples}"
            
            # Prepare training data
            X_new = np.array(features)
            y_new = [name] * len(features)
            
            # Load existing data if available
            if self.knn_model is not None:
                # Get existing training data
                existing_X = self.pca.inverse_transform(self.knn_model._fit_X)
                existing_y = [list(self.label_encoder.keys())[list(self.label_encoder.values()).index(label)] 
                             for label in self.knn_model._y]
                
                # Combine with new data
                X_combined = np.vstack([existing_X, X_new])
                y_combined = existing_y + y_new
            else:
                X_combined = X_new
                y_combined = y_new
            
            # Update label encoder
            unique_labels = list(set(y_combined))
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
            y_encoded = [self.label_encoder[label] for label in y_combined]
            
            # Apply PCA for dimensionality reduction
            n_components = min(50, len(X_combined) - 1, X_combined.shape[1])
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X_combined)
            
            # Train KNN model
            self.knn_model = KNeighborsClassifier(n_neighbors=min(5, len(X_pca)))
            self.knn_model.fit(X_pca, y_encoded)
            
            # Save models
            self.save_models()
            
            logger.info(f"Successfully added {valid_samples} samples for {name}")
            return True, f"Successfully registered {name} with {valid_samples} face samples"
            
        except Exception as e:
            logger.error(f"Error adding new face: {e}")
            return False, f"Error during registration: {str(e)}"
    
    def recognize_face(self, image_data):
        """Recognize a face from image data"""
        try:
            if self.knn_model is None or self.pca is None:
                return "Unknown", 0.0
            
            image = self.base64_to_image(image_data)
            if image is None:
                return "Unknown", 0.0
            
            faces, gray = self.detect_faces(image)
            
            if len(faces) == 0:
                return "No face detected", 0.0
            
            # Use the largest face detected
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_features = self.extract_face_features(gray, largest_face)
            
            # Apply PCA transformation
            face_pca = self.pca.transform([face_features])
            
            # Get prediction probabilities
            probabilities = self.knn_model.predict_proba(face_pca)[0]
            predicted_label = self.knn_model.predict(face_pca)[0]
            max_probability = max(probabilities)
            
            # Get name from label
            name = list(self.label_encoder.keys())[list(self.label_encoder.values()).index(predicted_label)]
            
            if max_probability < self.confidence_threshold:
                return "Unknown", max_probability
            
            return name, max_probability
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return "Error", 0.0

# Initialize the face recognition system
face_system = FaceRecognitionSystem()

@app.route('/register', methods=['POST'])
def register():
    """Register a new face"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data or 'images' not in data:
            return jsonify({'success': False, 'message': 'Missing name or images'}), 400
        
        name = data['name'].strip()
        images = data['images']
        
        if not name:
            return jsonify({'success': False, 'message': 'Name cannot be empty'}), 400
        
        if len(images) < face_system.min_samples:
            return jsonify({
                'success': False, 
                'message': f'Need at least {face_system.min_samples} images'
            }), 400
        
        success, message = face_system.add_new_face(name, images)
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'message': message}), 400
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Server error during registration'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Recognize a face for login"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Missing image data'}), 400
        
        image_data = data['image']
        name, probability = face_system.recognize_face(image_data)
        
        return jsonify({
            'success': True,
            'name': name,
            'probability': round(probability, 3)
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Server error during recognition'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': face_system.knn_model is not None})

if __name__ == '__main__':
    logger.info("Starting Face Recognition API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)