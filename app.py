# app.py

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

# Load the Facenet model
model = load_model('facenet_keras.h5')

# Initialize MTCNN face detector
detector = MTCNN()

app = Flask(__name__)

def extract_face(image, required_size=(160, 160)):
    """ Extract a single face from a given photograph """
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None, None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    image = cv2.resize(face, required_size)
    return image, results[0]['box']

def get_embedding(face):
    """ Get the face embedding using Facenet model """
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    samples = np.expand_dims(face, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

def save_embedding(username, embedding):
    """ Save the embedding to a file """
    os.makedirs('./embeddings', exist_ok=True)
    np.save(f'./embeddings/{username}.npy', embedding)

def load_embedding(username):
    """ Load the embedding from a file """
    return np.load(f'./embeddings/{username}.npy')

def capture_face_images(username, num_images=5):
    """ Capture multiple images of a user's face """
    cap = cv2.VideoCapture(0)
    count = 0
    embeddings = []
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        face, _ = extract_face(frame)
        if face is not None:
            embedding = get_embedding(face)
            embeddings.append(embedding)
            count += 1
        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    avg_embedding = np.mean(embeddings, axis=0)
    save_embedding(username, avg_embedding)

def verify_user(username, threshold=0.5):
    """ Verify the user by comparing the embedding """
    stored_embedding = load_embedding(username)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face, _ = extract_face(frame)
        if face is not None:
            embedding = get_embedding(face)
            dist = np.linalg.norm(stored_embedding - embedding)
            if dist < threshold:
                cap.release()
                cv2.destroyAllWindows()
                return True
        cv2.imshow('Login Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    capture_face_images(username)
    return jsonify({"message": f"User {username} registered successfully."})

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    if verify_user(username):
        return jsonify({"message": "Login successful!"})
    else:
        return jsonify({"message": "Login failed!"}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
