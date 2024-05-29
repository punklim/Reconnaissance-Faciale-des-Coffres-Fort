#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import cv2
import joblib
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Fonction pour capturer des images
def capture_images(label, num_images=15):
    cap = cv2.VideoCapture(0)
    images = []
    count = 0
    start_time = time.time()
    image_placeholder = st.empty()  # Placeholder pour les images
    while count < num_images and time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            images.append(face_img.flatten())
            count += 1
            image_placeholder.image(face_img, channels="GRAY")
        else:
            st.warning("Veuillez vous assurer qu'il n'y a qu'un seul visage visible.")
        if count == num_images:
            break
    cap.release()
    return images

# Fonction pour entraîner le modèle
def train_model(images, labels):
    if len(set(labels)) < 2:
        st.error("Le nombre de classes doit être supérieur à un pour entraîner le modèle.")
        return None
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = SVC(gamma='scale')
    clf.fit(images, y)
    joblib.dump((clf, le), "face_recognition_model.pkl")

# Fonction pour reconnaître le visage
def recognize_face():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Erreur lors de l'accès à la caméra."
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100)).flatten().reshape(1, -1)
        clf, le = joblib.load("face_recognition_model.pkl")
        y_pred = clf.predict(face_img)
        label = le.inverse_transform(y_pred)[0]
        if label == "known":
            return "Accès autorisé."
        else:
            return "Accès refusé."
    else:
        return "Veuillez vous assurer qu'il n'y a qu'un seul visage visible."

# Chargement du classificateur de visage Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Interface utilisateur Streamlit
st.title("Système de reconnaissance faciale pour coffre-fort")

menu = ["Entraînement", "Reconnaissance"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Entraînement":
    st.subheader("Entraînement du modèle")
    label = st.text_input("Nom de la personne", "known")
    if st.button("Capturer des images et entraîner le modèle"):
        with st.spinner("Capture d'images en cours..."):
            images = capture_images(label)
            st.success("Capture terminée.")
        with st.spinner("Capture des images pour classe 'unknown'..."):
            unknown_images = capture_images("unknown")
            st.success("Capture des images pour 'unknown' terminée.")
        images.extend(unknown_images)
        labels = [label] * len(images[:len(images)//2]) + ["unknown"] * len(images[len(images)//2:])
        with st.spinner("Entraînement du modèle..."):
            train_model(images, labels)
            st.success("Modèle entraîné et sauvegardé.")

elif choice == "Reconnaissance":
    st.subheader("Reconnaissance faciale")
    if st.button("Vérifier l'accès"):
        with st.spinner("Vérification en cours..."):
            result = recognize_face()
            st.success(result)
