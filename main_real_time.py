import os
import cv2
import numpy as np
import time
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ultralytics import YOLO
from sort import *
from OCR_real_time import *
from prettytable import PrettyTable  # Import PrettyTable to display results in a tabular format


# Constants
DATABASE_URL = 'postgresql://postgres:1995@localhost:5432/postgres'
VIDEO_FILE = 0
OUTPUT_FILE = 'output.avi'
FPS_DISPLAY_POSITION = (10, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS_COLOR = (100, 255, 0)
BOX_COLOR = (0, 255, 0)

# SQLAlchemy setup
Base = declarative_base()
class Result(Base):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    frame_number = Column(Integer)
    car_id = Column(Integer)
    license_plate_text = Column(String)
    text_score = Column(Float)
    timestamp = Column(DateTime)

def setup_database():
    """Initialize the database and create the results table if it doesn't exist."""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()

def load_models():
    """Load the YOLO models for vehicle detection and license plate recognition."""
    coco_model = YOLO('yolov8s.pt')
    license_plate_detector = YOLO('best.pt')
    return coco_model, license_plate_detector

def initialize_video_output(cap):
    """Initialize video writer for output video."""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))

def detect_vehicles(coco_model, frame):
    """Detect vehicles in the current frame using YOLO."""
    detections = coco_model(frame)[0]
    vehicles = [2, 3, 5, 7]
    return [[*detection[:5]] for detection in detections.boxes.data.tolist() if int(detection[5]) in vehicles]

def process_license_plate(license_plate_crop):
    """Process the cropped license plate image for text recognition."""
    gray_plate = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
    thresh_plate = cv2.adaptiveThreshold(blurred_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return read_license_plate(thresh_plate)
def display_results_table(session):
    """Fetch results from the database and display them in a table."""
    # Create a table with PrettyTable
    table = PrettyTable()
    table.field_names = ["ID", "Frame Number", "Car ID", "License Plate", "Text Score", "Timestamp"]

    # Fetch all results from the database
    results = session.query(Result).filter(Result.text_score > 0.8).all()

    # Add each result to the table
    for result in results:
        table.add_row([
            result.id,
            result.frame_number,
            result.car_id,
            result.license_plate_text,
            f"{result.text_score:.2f}",
            result.timestamp
        ])
    
    # Print the table to the console
    print(table)

def main():
    """Main function for license plate recognition."""
    
    
    # Database setup
    session = setup_database()

    # Load models
    coco_model, license_plate_detector = load_models()
    
    # Video capture
    cap = cv2.VideoCapture(VIDEO_FILE)
    out = initialize_video_output(cap)

    # Tracker initialization
    mot_tracker = Sort()
    frame_nmr = -1
    best_scores = {}
    final_scores = {}
    prev_frame_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1
        
        # Calculate and display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", FPS_DISPLAY_POSITION, FONT, 1, FPS_COLOR, 2, cv2.LINE_AA)

        # Detect vehicles
        detections_ = detect_vehicles(coco_model, frame)

        # Update tracker
        track_ids = mot_tracker.update(np.asarray(detections_)) if detections_ else np.array([])

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_text, license_plate_text_score = process_license_plate(license_plate_crop)

                if license_plate_text:
                    timestamp = datetime.now()
                    if car_id not in best_scores or license_plate_text_score > best_scores[car_id]['text_score']:
                        best_scores[car_id] = {
                            'frame_number': frame_nmr,
                            'license_plate_text': license_plate_text,
                            'text_score': license_plate_text_score,
                            'timestamp': timestamp
                        }
                    display_text = f"{license_plate_text} ({license_plate_text_score * 100:.1f}%)"
                    cv2.putText(frame, display_text, (int(x1), int(y1) - 10), FONT, 0.9, BOX_COLOR, 2)

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)

        cv2.imshow('License Plate Recognition', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Insertion des résultats dans la base de données
    for car_id, data in best_scores.items():
        license_plate_text = data['license_plate_text']
        text_score = data['text_score']
        timestamp = data['timestamp']

        if license_plate_text not in final_scores or text_score > final_scores[license_plate_text]['text_score']:
            final_scores[license_plate_text] = {
                'frame_number': data['frame_number'],
                'car_id': car_id,
                'license_plate_text': license_plate_text,
                'text_score': text_score,
                'timestamp': timestamp
            }

    for license_plate_text, data in final_scores.items():
        result = Result(
            frame_number=data['frame_number'],
            car_id=data['car_id'],
            license_plate_text=license_plate_text,
            text_score=data['text_score'],
            timestamp=data['timestamp']
        )
        session.add(result)

    session.commit()

    # Display the results table after processing
    display_results_table(session)
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
