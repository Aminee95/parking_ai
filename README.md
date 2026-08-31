<div align="center">

# 🚗 Automatic Number Plate Recognition (ANPR) System

Real-time license plate detection and multilingual recognition, built for smart parking and urban surveillance in Amsterdam.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-brightgreen)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Multilingual-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Data Collection](#data-collection)
6. [Model Training](#model-training)
7. [Database Setup](#database-setup)
8. [Getting Started](#getting-started)
9. [Docker Deployment](#docker-deployment)
10. [Results](#results)
11. [Roadmap](#roadmap)
12. [Contributing](#contributing)
13. [License](#license)

---

## Introduction

**ANPR** is a real-time **Automatic Number Plate Recognition** system designed to detect and read vehicle license plates from images or video streams. It was built to support **parking management and surveillance** use cases in urban environments such as **Amsterdam**, with a strong emphasis on **multilingual support** — handling both **Latin and Arabic** character plates.

The pipeline combines:
- **YOLOv8** for fast, accurate plate localization
- **EasyOCR** for multilingual text extraction
- **PostgreSQL + SQLAlchemy** for persistent, queryable storage of detections

The system is fully **containerized with Docker** and designed to run on **edge devices** such as the **Jetson Nano** for real-time, on-site deployment.

---

## Features

| Feature | Description |
|---|---|
| 🔍 Real-time detection | Plate localization using YOLOv8 |
| 🌍 Multilingual OCR | Latin & Arabic character recognition via EasyOCR |
| 🗄️ Persistent storage | PostgreSQL database with detection time & GPS coordinates |
| 🌐 Multi-country ready | Adaptable to different plate formats (NL, FR, and more) |
| 📦 Containerized | One-command deployment via Docker |
| ⚡ Edge-ready | Optimized for real-time inference on Jetson Nano |

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.8+ |
| Detection | YOLOv8 |
| OCR | EasyOCR |
| Dataset annotation | Roboflow (24,000 images) |
| ORM | SQLAlchemy |
| Database | PostgreSQL |
| Containerization | Docker |
| Edge deployment | NVIDIA Jetson Nano |

---

## Project Structure

```
anpr/
├── data/          # Datasets (raw and annotated)
├── models/        # Pre-trained and fine-tuned model weights
├── scripts/       # Detection and recognition scripts
├── database/       # SQLAlchemy models and DB configuration
├── docker/        # Docker configuration files
├── tests/         # Unit tests
├── results/       # Logs and evaluation outputs
└── README.md      # Project documentation
```

---

## Data Collection

The training dataset was collected and annotated using **[Roboflow](https://roboflow.com/)**, and consists of **24,000 annotated images** of license plates from multiple countries, covering a wide range of formats and lighting conditions.

| Split | Percentage |
|---|---|
| Training | 87% |
| Validation | 7% |
| Test | 4% |

---

## Model Training

**YOLOv8** handles plate detection, trained on a GPU-based environment using the Roboflow dataset. **EasyOCR** handles text recognition, chosen for its native support of both **Arabic and Latin** scripts.

### Train the detector

```bash
python train.py --data data.yaml --cfg cfg/yolov8.cfg --epochs 50 --weights yolov8.pt
```

### Test OCR on a single image

```bash
python ocr_test.py --image <image-path>
```

---

## Database Setup

Detected plates — along with detection timestamp and GPS coordinates — are persisted in **PostgreSQL**, accessed through **SQLAlchemy**.

### 1. Install PostgreSQL

```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

### 2. Create the database

```bash
sudo -i -u postgres
createdb anpr_db
```

### 3. Configure the connection string

Update your project configuration with your PostgreSQL credentials:

```bash
SQLALCHEMY_DATABASE_URI = 'postgresql://username:password@localhost/anpr_db'
```

### Example schema

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class PlateDetection(Base):
    __tablename__ = 'detected_plates'

    id = Column(Integer, primary_key=True)
    plate_number = Column(String, nullable=False)
    detection_time = Column(DateTime, default=datetime.utcnow)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

engine = create_engine('postgresql://username:password@localhost/anpr_db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
```

### Insert a record

```python
new_plate = PlateDetection(
    plate_number="ABC123",
    latitude=52.3676,
    longitude=4.9041
)
session.add(new_plate)
session.commit()
```

### Query records

```python
results = session.query(PlateDetection).all()
for plate in results:
    print(plate.plate_number, plate.detection_time, plate.latitude, plate.longitude)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL
- Docker *(optional, for containerized deployment)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/anpr.git
cd anpr

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run detection on an image or video feed
python detect.py --image <path-to-image> --save-db
```

Once detection runs, results are automatically stored in PostgreSQL and can be queried as shown above.

---

## Docker Deployment

Run the entire system in an isolated, reproducible environment.

```bash
# Build the image
docker build -t anpr-system .

# Run the container
docker run -d -p 8000:8000 anpr-system
```

The application will be available at **http://localhost:8000**.

---

## Results

| Metric | Score |
|---|---|
| Detection accuracy (YOLOv8) | **>95%** |
| Recognition accuracy (EasyOCR) | **~90%** (strongest on Latin characters) |
| Average processing time | **0.2s / frame** (real-time capable) |

---

## Roadmap

- [ ] Expand coverage to more countries (Europe & Middle East)
- [ ] Improve OCR accuracy on mixed Arabic/Latin plates
- [ ] Build a web-based dashboard / UI for live monitoring
- [ ] Full Jetson Nano integration for mobile, real-time deployment

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Open a pull request — we'll review it as soon as possible

---

## License

This project is licensed under the **[MIT License](./LICENSE)**.
