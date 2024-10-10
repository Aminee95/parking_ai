# 1. Utilisation d'une image de base Python
FROM python:3.8.10-slim

# 2. Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0-dev\
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app


COPY . /app

# 3. Copier les fichiers requirements.txt dans le conteneur
#COPY requirements.txt .

# 4. Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copier le module best.pt
#COPY best.pt .

# 5. Copier tout le contenu du projet dans le conteneur
#COPY . .

# Exposer un port 
#EXPOSE 8000

# 7. Définir la commande de démarrage
CMD ["python", "main_real_time.py"]
