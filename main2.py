import os
import psycopg2
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import *
from OCR import *
import time

# Changer le répertoire de travail vers le dossier du projet
os.chdir(r"C:\Users\amine\Desktop\projects\Reconnaissance_plaques")

# Dictionnaire pour stocker les résultats par frame
results = {}

# Dictionnaire pour stocker le meilleur score de texte par car_id
best_scores = {}

# Dictionnaire pour stocker le meilleur score de texte par license_plate_text
final_scores = {}

# Fonction pour comparer deux chaînes et compter les caractères similaires
def similar_characters(text1, text2):
    # Utiliser le nombre de caractères communs entre deux textes
    return sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)

# Initialiser le tracker d'objets avec l'algorithme SORT
mot_tracker = Sort()

# Charger les modèles
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')

# Charger la vidéo
cap = cv2.VideoCapture('a.mp4')

# Obtenir les propriétés de la vidéo
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialiser l'écrivain vidéo
output_file = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Liste des classes de véhicules à détecter
vehicles = [2, 3, 5, 7]

# Initialiser le calcul du FPS
prev_frame_time = 0

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    dbname='anpr_data',  # Remplacez par le nom de votre base de données
    user='postgres',      # Remplacez par votre nom d'utilisateur PostgreSQL
    password='1995',      # Remplacez par votre mot de passe
    host='localhost',     # Ou l'adresse de votre serveur PostgreSQL
    port='5432'           # Port par défaut de PostgreSQL
)
c = conn.cursor()

# Lecture des frames de la vidéo
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = np.array([])

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
                thresh_plate = cv2.adaptiveThreshold(blurred_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                license_plate_text, license_plate_text_score = read_license_plate(thresh_plate)

                if license_plate_text is not None:
                    timestamp = datetime.now()
                    
                    # Mettre à jour le meilleur score pour chaque car_id
                    if car_id not in best_scores or license_plate_text_score > best_scores[car_id]['text_score']:
                        best_scores[car_id] = {
                            'frame_number': frame_nmr,
                            'license_plate_text': license_plate_text,
                            'text_score': license_plate_text_score,
                            'timestamp': timestamp
                        }

                    display_text = f"{license_plate_text} ({license_plate_text_score * 100:.1f}%)"
                    cv2.putText(frame, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Dessiner des boîtes englobantes pour les plaques d'immatriculation détectées
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Dessiner la boîte en vert

        cv2.imshow('License Plate Recognition', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Insertion des meilleurs scores par car_id dans best_scores
for car_id, data in best_scores.items():
    license_plate_text = data['license_plate_text']
    text_score = data['text_score']
    timestamp = data['timestamp']
    
    # Mettre à jour le meilleur score par license_plate_text
    if license_plate_text not in final_scores or text_score > final_scores[license_plate_text]['text_score']:
        final_scores[license_plate_text] = {
            'frame_number': data['frame_number'],
            'car_id': car_id,
            'license_plate_text': license_plate_text,
            'text_score': text_score,
            'timestamp': timestamp
        }

# Nouvelle étape : Filtrer par similarité de caractères (au moins 5 caractères similaires)
filtered_scores = {}
for license_plate_text1, data1 in final_scores.items():
    keep_data = True
    for license_plate_text2, data2 in final_scores.items():
        if license_plate_text1 != license_plate_text2:
            # Vérifier si les deux plaques ont au moins 5 caractères similaires
            if similar_characters(license_plate_text1, license_plate_text2) >= 5:
                # Conserver celle avec le score le plus élevé
                if data1['text_score'] < data2['text_score']:
                    keep_data = False
                    break
    if keep_data:
        filtered_scores[license_plate_text1] = data1

# Insertion des résultats filtrés dans la base de données
for license_plate_text, data in filtered_scores.items():
    c.execute('''
    INSERT INTO results (frame_number, car_id, license_plate_text, text_score, timestamp)
    VALUES (%s, %s, %s, %s, %s)
    ''', (data['frame_number'], data['car_id'], license_plate_text, data['text_score'], data['timestamp']))
conn.commit()

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()

# Fermer la connexion à la base de données
conn.close()
