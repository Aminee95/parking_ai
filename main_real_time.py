import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import *
from OCR import *
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Changer le répertoire de travail vers le dossier du projet
os.chdir(r"C:\Users\amine\Desktop\projects\Reconnaissance_plaques")

# Configuration de SQLAlchemy
DATABASE_URL = 'postgresql://postgres:1995@localhost:5432/postgres'

# Créer l'engine de connexion à la base de données
engine = create_engine(DATABASE_URL)

# Créer une session de base de données
Session = sessionmaker(bind=engine)
session = Session()

# Déclaration de la base de données avec SQLAlchemy ORM
Base = declarative_base()

class Result(Base):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_number = Column(Integer)
    car_id = Column(Integer)
    license_plate_text = Column(String)
    text_score = Column(Float)
    timestamp = Column(DateTime)

# Créer la table dans la base de données si elle n'existe pas déjà
Base.metadata.create_all(engine)

# Initialiser le tracker d'objets avec l'algorithme SORT (pour le suivi des véhicules)
mot_tracker = Sort()

# Charger le modèle COCO (pour la détection des véhicules)
coco_model = YOLO('yolov8n.pt')

# Charger le modèle personnalisé pour la détection des plaques d'immatriculation
license_plate_detector = YOLO('best.pt')

# Charger la vidéo pour traitement
cap = cv2.VideoCapture(0)

# Obtenir les propriétés de la vidéo : FPS (images par seconde), largeur et hauteur des frames
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialiser l'écrivain vidéo pour sauvegarder la sortie avec boîtes englobantes et texte
output_file = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Liste des classes de véhicules à détecter (basé sur COCO, par exemple voiture, bus, etc.)
vehicles = [2, 3, 5, 7]

# Initialiser le temps pour calculer le FPS (images par seconde)
prev_frame_time = 0

# Initialiser le numéro de frame (compteur de frames)
frame_nmr = -1

# Dictionnaire pour stocker les résultats par frame
results = {}

# Dictionnaire pour stocker le meilleur score de texte par car_id (FILTRE 1)
best_scores = {}

# Dictionnaire pour stocker le meilleur score de texte par license_plate_text (FILTRE 2)
final_scores = {}

# Lire les frames de la vidéo
ret = True
while ret:
    frame_nmr += 1  # Incrémenter le numéro de frame à chaque itération
    ret, frame = cap.read()  # Lire une frame de la vidéo
    if ret:  # Si la frame est valide
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)  # Calculer le FPS en temps réel
        prev_frame_time = new_frame_time
        fps = int(fps)

        # Afficher le FPS sur la vidéo
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        # Détection des véhicules dans la frame avec YOLO (modèle COCO)
        detections = coco_model(frame)[0]
        detections_ = []  # Liste pour stocker les détections de véhicules valides
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection  # Coordonnées et score de la boîte
            if int(class_id) in vehicles:  # Filtrer les véhicules d'intérêt (basé sur leur ID)
                detections_.append([x1, y1, x2, y2, score])  # Ajouter la boîte à la liste des détections

        # Mise à jour du tracker SORT si des véhicules sont détectés
        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = np.array([])  # Aucune détection de véhicule

        # Détection des plaques d'immatriculation dans la frame
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate  # Extraire les coordonnées de la boîte et le score
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)  # Associer la plaque à une voiture

            if car_id != -1:  # Si une voiture est associée à la plaque
                # Extraire la zone de la plaque d'immatriculation
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
                blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)  # Appliquer un flou pour réduire le bruit
                thresh_plate = cv2.adaptiveThreshold(blurred_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Lire le texte de la plaque avec OCR (Optical Character Recognition)
                license_plate_text, license_plate_text_score = read_license_plate(thresh_plate)

                if license_plate_text is not None:  # Si du texte a été détecté
                    timestamp = datetime.now()  # Obtenir l'horodatage actuel
                    
                    # FILTRE 1: Mettre à jour le meilleur score pour chaque car_id
                    if car_id not in best_scores or license_plate_text_score > best_scores[car_id]['text_score']:
                        best_scores[car_id] = {
                            'frame_number': frame_nmr,
                            'license_plate_text': license_plate_text,
                            'text_score': license_plate_text_score,
                            'timestamp': timestamp
                        }

                    # Afficher le texte reconnu sur la vidéo avec le score
                    display_text = f"{license_plate_text} ({license_plate_text_score * 100:.1f}%)"
                    cv2.putText(frame, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Dessiner les boîtes englobantes autour des plaques détectées
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Dessiner une boîte verte

        # Afficher la vidéo avec les boîtes englobantes
        cv2.imshow('License Plate Recognition', frame)
        out.write(frame)  # Sauvegarder la frame dans la vidéo de sortie

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Si l'utilisateur appuie sur 'q', quitter la boucle
            break

# FILTRE 2: Insertion uniquement des meilleurs scores pour chaque license_plate_text
for car_id, data in best_scores.items():
    license_plate_text = data['license_plate_text']
    text_score = data['text_score']
    timestamp = data['timestamp']
    
    # Mettre à jour le meilleur score pour chaque texte similaire de plaque
    if license_plate_text not in final_scores or text_score > final_scores[license_plate_text]['text_score']:
        final_scores[license_plate_text] = {
            'frame_number': data['frame_number'],
            'car_id': car_id,
            'license_plate_text': license_plate_text,
            'text_score': text_score,
            'timestamp': timestamp
        }

# Insertion des meilleurs scores dans la base de données avec SQLAlchemy
for license_plate_text, data in final_scores.items():
    result = Result(
        frame_number=data['frame_number'],
        car_id=data['car_id'],
        license_plate_text=license_plate_text,
        text_score=data['text_score'],
        timestamp=data['timestamp']
    )
    session.add(result)

session.commit()  # Sauvegarder les modifications dans la base de données

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
