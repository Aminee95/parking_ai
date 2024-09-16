# Système de Reconnaissance Automatique de Plaques d'Immatriculation (ANPR) en Temps Réel

## Description
Ce projet implémente un système de reconnaissance automatique des plaques d'immatriculation (ANPR) utilisant des modèles d'apprentissage profond (YOLO pour la détection d'objets) et EasyOCR pour la reconnaissance de texte. Le système est conçu pour fonctionner en temps réel et est destiné à la gestion des parkings ou à la surveillance des véhicules. Les plaques d'immatriculation sont détectées et lues à partir de vidéos ou de flux en direct, puis stockées dans une base de données pour un traitement ultérieur.

## Fonctionnalités
- Détection des véhicules et des plaques d'immatriculation en temps réel à partir de vidéos.
- Lecture et reconnaissance de texte des plaques en plusieurs langues (anglais, arabe, etc.).
- Traque des véhicules via un algorithme de suivi (SORT).
- Filtrage et sélection du meilleur score de reconnaissance de texte.
- Stockage des données dans une base de données PostgreSQL.

## Prérequis

### Logiciels
- Python 3.8 
- OpenCV
- EasyOCR
- YOLO (You Only Look Once)
- NumPy
- psycopg2 (pour la connexion à PostgreSQL)
- SORT (Simple Online and Realtime Tracking)

### Matériel
- Nvidia Jetson Nano 
- Caméra compatible (USB ou CSI)
- Base de données PostgreSQL

- 
## Installation

1. Clonez le dépôt du projet :
```bash
git clone https://github.com/Aminee95/parking_ai.git

cd reconnaissance_plaque
pip install -r requirements.txt

