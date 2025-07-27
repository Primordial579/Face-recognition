import cv2
import numpy as np
import os
import pickle
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC

# Constants
MODEL_FILE = "face_recognition_enhanced.pkl"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
SAMPLES_PER_PERSON = 400 # 200 original + 200 flipped

def initialize_model():
    """Initialize or load model with name mapping"""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['scaler'], model_data['kpca'], model_data['svm'], model_data['name_map']
    else:
        # Load base dataset
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X, y = faces.data, faces.target
        
        # Initialize components
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kpca = KernelPCA(n_components=200, kernel="rbf", gamma=0.03, fit_inverse_transform=True, random_state=42)
        X_kpca = kpca.fit_transform(X_scaled)
        
        svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
        svm.fit(X_kpca, y)
        
        # Create default name map
        name_map = {i: f"Person {i}" for i in range(40)}
        
        save_model(scaler, kpca, svm, name_map)
        return scaler, kpca, svm, name_map

def save_model(scaler, kpca, svm, name_map):
    """Save model with name mapping"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'kpca': kpca,
            'svm': svm,
            'name_map': name_map
        }, f)

def capture_face_samples():
    """Capture 600 samples (300 original + 300 flipped) with name input"""
    name = input("Enter name for new face: ").strip()
    if not name:
        print("Name cannot be empty!")
        return None, None
    
    cap = cv2.VideoCapture(0)
    samples = []
    print(f"\nCapturing {SAMPLES_PER_PERSON} samples for {name}...")
    print("Please move your head slowly in different directions")

    while len(samples) < SAMPLES_PER_PERSON:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced face detection
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            # Extract and enhance face
            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)
            
            # Corrected gamma correction
            gamma = 1.7
            table = np.array([(i / 255.0) ** (1 / gamma) * 255 for i in range(256)]).astype("uint8")

            face = cv2.LUT(face, table)
            
            # Add original and flipped samples
            resized_face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_CUBIC).flatten()
            samples.append(resized_face)
            
            flipped_face = cv2.resize(cv2.flip(face, 1), (64, 64), interpolation=cv2.INTER_CUBIC).flatten()
            samples.append(flipped_face)
            
            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {len(samples)//2}/{SAMPLES_PER_PERSON//2}", 
                       (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f"Registering: {name} (Press Q to finish)", frame)
        if cv2.waitKey(30) & 0xFF == ord('q') or len(samples) >= SAMPLES_PER_PERSON:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return np.array(samples), name

def add_new_face():
    """Register new face with 600 samples"""
    samples, name = capture_face_samples()
    if samples is None:
        return
    
    scaler, kpca, svm, name_map = initialize_model()
    
    # Get next available label
    new_label = max(name_map.keys()) + 1 if name_map else 0
    
    # Combine with existing data
    X_combined = np.vstack([kpca.inverse_transform(svm.support_vectors_), samples])
    y_combined = np.concatenate([svm.predict(svm.support_vectors_), [new_label]*len(samples)])
    
    # Update model
    X_scaled = scaler.fit_transform(X_combined)
    X_kpca = kpca.fit_transform(X_scaled)
    svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm.fit(X_kpca, y_combined)
    
    # Update name map
    name_map[new_label] = name
    save_model(scaler, kpca, svm, name_map)
    print(f"\nSuccessfully registered {name} with {SAMPLES_PER_PERSON} samples!")

def live_recognition():
    """Real-time recognition with names"""
    scaler, kpca, svm, name_map = initialize_model()
    cap = cv2.VideoCapture(0)
    CONFIDENCE_THRESHOLD = 0.88
    
    print("\nStarting live recognition (Press Q to quit)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            # Advanced preprocessing
            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)
            face = cv2.GaussianBlur(face, (3, 3), 0)
            resized_face = cv2.resize(face, (64, 64), interpolation=cv2.INTER_CUBIC).flatten()
            
            # Prediction
            face_scaled = scaler.transform([resized_face])
            face_kpca = kpca.transform(face_scaled)
            proba = svm.predict_proba(face_kpca)[0]
            max_proba = np.max(proba)
            pred = svm.predict(face_kpca)[0] if max_proba > CONFIDENCE_THRESHOLD else -1
            
            # Display results
            display_name = name_map.get(pred, "Unknown") if pred != -1 else "Unknown (Low Confidence)"
            color = (0, 255, 0) if pred != -1 else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{display_name} ({max_proba:.1%})", 
                       (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow("Face Recognition - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    """Interactive menu system"""
    while True:
        print("\n===== Advanced Face Recognition =====")
        print("1. Register New Face (300 samples)")
        print("2. Live Face Recognition")
        print("3. Exit")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            add_new_face()
        elif choice == '2':
            live_recognition()
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main_menu()