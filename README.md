# Automatic Number Plate Recognition (ANPR) System

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Project Structure](#project-structure)
5. [Data Collection](#data-collection)
6. [Model Training](#model-training)
7. [Database Setup](#database-setup)
8. [Running the Project](#running-the-project)
9. [Using Docker](#using-docker)
10. [Results and Analysis](#results-and-analysis)
11. [Future Work](#future-work)
12. [Contributing](#contributing)
13. [License](#license)

## Introduction
This project is an **Automatic Number Plate Recognition (ANPR)** system designed to detect and recognize vehicle license plates in real-time. The project aims to improve parking management and surveillance, particularly in urban areas like **Amsterdam**, by using cutting-edge computer vision and machine learning techniques.

The system is capable of detecting number plates from various countries, including those using both Arabic and Latin characters. It leverages **YOLO** for number plate detection and **EasyOCR** for multilingual text recognition.

## Features
- Real-time number plate detection using YOLOv8.
- Multilingual number plate recognition with EasyOCR.
- Integration with a PostgreSQL database for storing plate numbers, detection time, and GPS coordinates.
- Flexible design that allows adaptation for multiple countries' number plates (Netherlands, France, etc.).
- Dockerized for easy deployment and testing.

## Technologies Used
- **Python 3.8+**: Main programming language.
- **YOLOv8**: For real-time object detection.
- **EasyOCR**: Multilingual Optical Character Recognition (OCR).
- **Roboflow**: Platform for dataset collection and annotation (24,000 images).
- **SQLAlchemy**: ORM for PostgreSQL database interaction.
- **PostgreSQL**: Database for storing detected license plate data.
- **Docker**: For containerizing the application and its dependencies.
- **Jetson Nano**: Embedded system for real-time edge deployment.

## Project Structure
 ├── data/ # Directory for datasets ├── models/ # Pre-trained and trained models ├── scripts/ # Python scripts for detection and recognition ├── database/ # SQLAlchemy models and database configurations ├── docker/ # Docker configurations for running the app ├── tests/ # Unit tests for the project ├── results/ # Logs and output from tests └── README.md # This readme file


## Data Collection
The dataset used for this project was collected and annotated using **Roboflow**. It consists of **24,000 images** of license plates from various countries, annotated with bounding boxes.

- **Training set**: 87%
- **Validation set**: 7%
- **Test set**: 4%

The dataset includes images from different countries with varying plate formats to ensure the model is robust and capable of detecting multiple formats.

## Model Training
- **YOLOv8** was used to train the model for number plate detection. The training was performed on a GPU-based environment, leveraging the dataset from Roboflow.
- For recognition, **EasyOCR** was employed due to its ability to handle multiple languages (Arabic and Latin characters).

### Training Commands:
```bash
# For YOLOv8 training
python train.py --data data.yaml --cfg cfg/yolov8.cfg --epochs 50 --weights yolov8.pt

# For OCR testing
python ocr_test.py --image <image-path>
```
Database Setup
The detected license plates, along with additional metadata like detection time and GPS coordinates, are stored in a PostgreSQL database. This data is managed using SQLAlchemy, which simplifies the process of interacting with the database.

Setting Up PostgreSQL
1. **Install PostgreSQL**:
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```
2. **Create a PostgreSQL Database: After installation, create a new database for the project**:
```bash
sudo -i -u postgres
createdb anpr_db
```
3. **Configure Database Access: Update the SQLAlchemy connection string with your PostgreSQL credentials in the project configuration file**:
```bash
SQLALCHEMY_DATABASE_URI = 'postgresql://username:password@localhost/anpr_db'
```
Using SQLAlchemy for Database Management
With SQLAlchemy, you can define database tables using Python classes, simplifying database operations.

Example of a table schema for storing detected plates:
```python
from sqlalchemy import create_engine, Column, String, DateTime, Float
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

# Create the database and table
engine = create_engine('postgresql://username:password@localhost/anpr_db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

Inserting a record:
```python
new_plate = PlateDetection(
    plate_number="ABC123",
    latitude=52.3676,
    longitude=4.9041
)
session.add(new_plate)
session.commit()

Querying records:
```python
results = session.query(PlateDetection).all()
for plate in results:
    print(plate.plate_number, plate.detection_time, plate.latitude, plate.longitude)
Running the Project
Prerequisites
Make sure you have the following installed:

Python 3.8+
PostgreSQL
Docker (optional for containerization)
