import os
os.chdir(r"C:\Users\amine\Desktop\projects\Reconnaissance_plaques")  # Changer le répertoire de travail vers le dossier du projet
from ultralytics import YOLO
import cv2

from datetime import datetime 
from sort import *
from OCR import *
import time

# Dictionnaire pour stocker les résultats
results = {}

# Initialiser le tracker d'objets avec l'algorithme SORT
mot_tracker = Sort()

# Charger les modèles
coco_model = YOLO('yolov8n.pt')  # Modèle YOLO pré-entraîné sur COCO pour la détection des véhicules
license_plate_detector = YOLO('best.pt')  # Modèle YOLO personnalisé pour la détection des plaques d'immatriculation



# Charger la vidéo
cap = cv2.VideoCapture('a.mp4')  # Charger la vidéo à traiter 



# Obtenir les propriétés de la vidéo pour sauvegarder la sortie
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtenir le nombre d'images par seconde de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtenir la largeur des images de la vidéo
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtenir la hauteur des images de la vidéo

# Initialiser l'écrivain vidéo
output_file = 'output.avi'  # Nom du fichier de sortie vidéo
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec vidéo utilisé pour compresser la vidéo
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))  # Initialiser l'écriture de la vidéo

# Liste des classes de véhicules à détecter (voiture,moto,bus, camion)
vehicles = [2, 3, 5, 7]

# Initialiser le calcul du FPS
prev_frame_time = 0


# Lecture des frames de la vidéo
frame_nmr = -1  # Compteur d'images
ret = True  # Variable pour contrôler la boucle de lecture des images
while ret:
    frame_nmr += 1  # Incrémenter le compteur d'images
    ret, frame = cap.read()  # Lire une image de la vidéo
    if ret:
        results[frame_nmr] = {}  # Initialiser un dictionnaire pour stocker les résultats de l'image actuelle
        # Calculer le temps actuel
        new_frame_time = time.time()
        
        # Calculer le FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Arrondir le FPS
        fps = int(fps)
        # Afficher le FPS sur l'image
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
        # Détecter les véhicules dans l'image
        detections = coco_model(frame)[0]  # Utiliser le modèle YOLO pour détecter les objets dans l'image
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection  # Extraire les valeurs
            if int(class_id) in vehicles:  # Vérifier si l'objet détecté est un véhicule
                detections_.append([x1, y1, x2, y2, score])  # Ajouter la détection à la liste

        # Vérifiez que detections_ n'est pas vide
        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.asarray(detections_))  # Mettre à jour le tracker avec les nouvelles détections
        else:
            track_ids = np.array([])  # Pas de détections


        # Détecter les plaques d'immatriculation
        license_plates = license_plate_detector(frame)[0]  # Utiliser le modèle personnalisé pour détecter les plaques d'immatriculation
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate  # Extraire les coordonnées et le score de la boîte englobante de la plaque

            # Associer la plaque d'immatriculation à une voiture
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)  # Trouver le véhicule correspondant à la plaque

            if car_id != -1:  # Si une correspondance est trouvée
                
                # Recadrer la plaque d'immatriculation
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]  # Extraire la région de l'image contenant la plaque
                # Convertir l'image en niveaux de gris
                gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                # Appliquer un flou pour réduire le bruit
                blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)

                # Appliquer un seuillage adaptatif
                thresh_plate = cv2.adaptiveThreshold(blurred_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
       

                # Lire le texte de la plaque d'immatriculationq
                license_plate_text, license_plate_text_score = read_license_plate(thresh_plate)  # Reconnaître le texte sur la plaque
                # Debug: Afficher le texte reconnu et le score
                
                if license_plate_text is not None:  # Si le texte est reconnu
                    timestamp = datetime.now()
                    results[frame_nmr][car_id] = {  # Sauvegarder les résultats pour ce véhicule
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},  # Boîte englobante du véhicule
                        'license_plate': {'bbox': [x1, y1, x2, y2],  # Boîte englobante de la plaque
                                          'text': license_plate_text,  # Texte reconnu
                                          'bbox_score': score,  # Score de détection de la plaque
                                          'text_score': license_plate_text_score} # Score de reconnaissance du texte
                                          
                                          
                    }
                    # Afficher le texte de la plaque et l'accuracy sur la vidéo
                    display_text = f"{license_plate_text} ({license_plate_text_score*100:.1f}%)"
                    cv2.putText(frame, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Dessiner des boîtes englobantes pour les véhicules détectés
        for detection in detections_:
            x1, y1, x2, y2, score = detection
        #   cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0 , 0), 2)  # Dessiner la boîte en vert

        # Dessiner des boîtes englobantes pour les plaques d'immatriculation détectées
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Dessiner la boîte en vert

        # Afficher l'image traitée
        cv2.imshow('License Plate Recognition', frame)

        # Écrire l'image dans le fichier vidéo de sortie
        out.write(frame)

        # Sortir de la boucle si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libérer les ressources
cv2.destroyAllWindows()  # Fermer toutes les fenêtres ouvertes

# Écrire les résultats dans un fichier CSV
write_csv(results, 'test.csv')  # Sauvegarder les résultats dans un fichier CSV
