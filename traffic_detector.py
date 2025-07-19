import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import json
from datetime import datetime
import sqlite3
import os


class TrafficViolationDetector:
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.3):
        """
        Initialize the traffic violation detection system with better settings
        """
        try:
            self.model = YOLO(model_path)
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            # Try alternative model loading
            try:
                import torch
                if hasattr(torch.serialization, 'add_safe_globals'):
                    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                self.model = YOLO(model_path)
                print("✅ YOLO model loaded with alternative method")
            except Exception as e2:
                print(f"❌ Still failed to load YOLO: {e2}")
                raise e2
        
        self.confidence_threshold = confidence_threshold
        
        # Enhanced tracking
        self.track_history = defaultdict(lambda: [])
        self.violations = []
        self.processed_violations = set()
        
        # Vehicle counting
        self.unique_vehicles = set()
        self.vehicle_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                               5: 'bus', 7: 'truck'}  # COCO classes
        self.total_detections = 0
        
        # Traffic light state detection
        self.traffic_light_state = "green"
        self.last_light_check = time.time()
        
        # Define violation zones
        self.stop_line_y = 300
        self.speed_zones = [(100, 200, 400, 300)]
        self.speed_limit = 30  # Lower for demo purposes
        
        # Enhanced license plate recognition
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            self.ocr_available = True
            print("✅ EasyOCR initialized for license plate recognition")
        except ImportError:
            print("⚠️  EasyOCR not available. Install with: pip install easyocr")
            self.ocr_available = False
        except Exception as e:
            print(f"⚠️  EasyOCR error: {e}")
            self.ocr_available = False
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing violations"""
        try:
            # Use absolute path to avoid issues
            database_path = os.path.abspath('traffic_violations.db')
            self.conn = sqlite3.connect(database_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            # Create enhanced violations table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    violation_type TEXT,
                    vehicle_id INTEGER,
                    vehicle_license_plate TEXT,
                    vehicle_type TEXT,
                    confidence REAL,
                    image_path TEXT,
                    video_path TEXT,
                    video_id INTEGER,
                    speed REAL,
                    location TEXT,
                    severity TEXT DEFAULT 'medium',
                    fine_amount REAL DEFAULT 100.0,
                    is_resolved BOOLEAN DEFAULT FALSE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''')
            
            self.conn.commit()
            print("✅ Traffic detector database initialized")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
    
    def detect_traffic_light_state(self, frame):
        """
        Enhanced traffic light detection using color analysis
        """
        # Simple color-based detection for demo
        # Define region of interest for traffic light (adjust coordinates)
        roi_x, roi_y, roi_w, roi_h = 50, 50, 100, 200
        
        if frame.shape[0] <= roi_y + roi_h or frame.shape[1] <= roi_x + roi_w:
            return self.traffic_light_state
        
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        if roi.size == 0:
            return self.traffic_light_state
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for traffic lights
            red_lower = np.array([0, 50, 50])
            red_upper = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            
            green_lower = np.array([40, 50, 50])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Count pixels for each color
            red_pixels = cv2.countNonZero(red_mask)
            green_pixels = cv2.countNonZero(green_mask)
            
            # Determine dominant color
            if red_pixels > 100:
                self.traffic_light_state = "red"
            elif green_pixels > 100:
                self.traffic_light_state = "green"
            
        except Exception as e:
            print(f"⚠️  Traffic light detection error: {e}")
        
        return self.traffic_light_state
    
    def calculate_speed(self, track_id, current_position, frame_time):
        """
        Enhanced speed calculation with smoothing
        """
        if track_id not in self.track_history:
            return 0
        
        history = self.track_history[track_id]
        if len(history) < 3:  # Need at least 3 points for stable calculation
            return 0
        
        try:
            # Use last 3 positions for smoother calculation
            recent_positions = history[-3:]
            total_distance = 0
            total_time = 0
            
            for i in range(1, len(recent_positions)):
                prev_pos, prev_time = recent_positions[i-1]
                curr_pos, curr_time = recent_positions[i]
                
                pixel_distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                       (curr_pos[1] - prev_pos[1])**2)
                
                # Convert to real distance (calibration needed)
                real_distance = pixel_distance * 0.1  # meters per pixel
                time_diff = curr_time - prev_time
                
                if time_diff > 0:
                    total_distance += real_distance
                    total_time += time_diff
            
            if total_time > 0:
                speed = (total_distance / total_time) * 3.6  # km/h
                return max(0, min(speed, 200))  # Cap at reasonable speed
            
        except Exception as e:
            print(f"⚠️  Speed calculation error: {e}")
        
        return 0
    
    def detect_license_plate(self, frame, bbox):
        """Enhanced license plate detection from vehicle bounding box"""
        if not self.ocr_available:
            return None
        
        try:
            if len(bbox) == 4:
                x, y, w, h = bbox
            else:
                return None
            
            # Extract vehicle region with some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            vehicle_region = frame[y1:y2, x1:x2]
            
            if vehicle_region.size == 0:
                return None
            
            # Focus on bottom part of vehicle (where plates usually are)
            plate_region_height = max(1, h // 3)
            plate_region = vehicle_region[-plate_region_height:, :]
            
            if plate_region.size == 0:
                return None
            
            # Enhance image for better OCR
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            # Apply image enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # OCR to detect text
            results = self.ocr_reader.readtext(enhanced, detail=0, paragraph=False)
            
            # Filter results for license plate patterns
            for text in results:
                if isinstance(text, str):
                    # Clean the text
                    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                    
                    # Basic license plate validation (adjust for your region)
                    if 4 <= len(cleaned_text) <= 10 and any(c.isdigit() for c in cleaned_text):
                        return cleaned_text
            
            return None
            
        except Exception as e:
            # Don't print every OCR error to avoid spam
            return None
    
    def calculate_fine_amount(self, violation_type, speed=None):
        """Calculate fine amount based on violation type"""
        fine_amounts = {
            'red_light_violation': 150.0,
            'speed_violation': 100.0 + (max(0, (speed or 50) - 30) * 5),
            'wrong_way_violation': 200.0,
            'no_helmet_violation': 75.0,
            'seatbelt_violation': 100.0,
            'phone_usage_violation': 125.0
        }
        return fine_amounts.get(violation_type, 100.0)
    
    def determine_severity(self, violation_type, speed=None):
        """Determine violation severity"""
        if violation_type in ['wrong_way_violation', 'red_light_violation']:
            return 'high'
        elif violation_type == 'speed_violation' and speed and speed > 60:
            return 'high'
        elif violation_type in ['phone_usage_violation', 'seatbelt_violation']:
            return 'medium'
        else:
            return 'low'
    
    def check_red_light_violation(self, detections, frame):
        """Enhanced red light violation detection - realistic simulation"""
        violations = []
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue
                
            bbox = detection['bbox']
            vehicle_bottom = bbox[1] + bbox[3]
            
            # Create unique violation key to prevent duplicates
            violation_key = f"red_light_{track_id}_{int(time.time()/10)}"
            
            # Simulate red light violation detection (every 12th vehicle for demo)
            if (vehicle_bottom > self.stop_line_y and 
                violation_key not in self.processed_violations and
                track_id % 12 == 0):
                
                self.processed_violations.add(violation_key)
                
                violation = {
                    'type': 'red_light_violation',
                    'track_id': track_id,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': detection['confidence'],
                    'bbox': bbox,
                    'severity': 'high',
                    'description': 'Vehicle crossed stop line during red light',
                    'vehicle_type': detection.get('vehicle_type', 'unknown')
                }
                violations.append(violation)
                print(f"🚨 Red light violation detected: Vehicle {track_id}")
        
        return violations
    
    def check_speed_violation(self, detections, frame_time):
        """Enhanced speed violation detection - more realistic"""
        violations = []
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue
            
            bbox = detection['bbox']
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            current_position = (center_x, center_y)
            
            # Update tracking history
            self.track_history[track_id].append((current_position, frame_time))
            
            # Keep only recent history
            if len(self.track_history[track_id]) > 15:
                self.track_history[track_id] = self.track_history[track_id][-15:]
            
            # Calculate speed
            speed = self.calculate_speed(track_id, current_position, frame_time)
            
            # Enhanced speed violation detection
            if speed > self.speed_limit and len(self.track_history[track_id]) >= 5:
                violation_key = f"speed_{track_id}_{int(frame_time/5)}"
                
                if violation_key not in self.processed_violations:
                    self.processed_violations.add(violation_key)
                    
                    severity = 'high' if speed > self.speed_limit * 2 else 'medium'
                    
                    violation = {
                        'type': 'speed_violation',
                        'track_id': track_id,
                        'speed': speed,
                        'speed_limit': self.speed_limit,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': detection['confidence'],
                        'bbox': bbox,
                        'severity': severity,
                        'description': f'Speed {speed:.1f} km/h exceeds limit {self.speed_limit} km/h',
                        'vehicle_type': detection.get('vehicle_type', 'unknown')
                    }
                    violations.append(violation)
                    print(f"🏃 Speed violation detected: Vehicle {track_id} at {speed:.1f} km/h")
        
        return violations
    
    def check_wrong_way_driving(self, detections):
        """Enhanced wrong way driving detection"""
        violations = []
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None or len(self.track_history[track_id]) < 5:
                continue
            
            # Analyze movement direction
            history = self.track_history[track_id]
            if len(history) < 5:
                continue
            
            # Calculate overall direction over last 5 positions
            start_pos = history[-5][0]
            end_pos = history[-1][0]
            
            direction_x = end_pos[0] - start_pos[0]
            direction_y = end_pos[1] - start_pos[1]
            
            # For demo: vehicles moving significantly upward are "wrong way"
            if direction_y < -30 and abs(direction_x) < 50:
                violation_key = f"wrong_way_{track_id}"
                
                if violation_key not in self.processed_violations:
                    self.processed_violations.add(violation_key)
                    
                    violation = {
                        'type': 'wrong_way_violation',
                        'track_id': track_id,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': detection['confidence'],
                        'bbox': detection['bbox'],
                        'severity': 'high',
                        'description': 'Vehicle driving against traffic flow',
                        'vehicle_type': detection.get('vehicle_type', 'unknown')
                    }
                    violations.append(violation)
                    print(f"↩️ Wrong way violation detected: Vehicle {track_id}")
        
        return violations
    
    def check_safety_violations(self, detections, frame):
        """Enhanced safety violation detection - simulated for demo"""
        violations = []
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue
            
            # Simulate safety violations for demo (every 25th vehicle)
            if track_id % 25 == 0:
                violation_key = f"safety_{track_id}"
                
                if violation_key not in self.processed_violations:
                    self.processed_violations.add(violation_key)
                    
                    # Determine violation type based on vehicle type
                    vehicle_type = detection.get('vehicle_type', 'car')
                    if vehicle_type == 'motorcycle':
                        violation_type = 'no_helmet_violation'
                        description = 'Motorcycle rider not wearing helmet'
                    else:
                        violation_type = 'seatbelt_violation'
                        description = 'Driver not wearing seatbelt'
                    
                    violation = {
                        'type': violation_type,
                        'track_id': track_id,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': detection['confidence'],
                        'bbox': detection['bbox'],
                        'severity': 'medium',
                        'description': description,
                        'vehicle_type': vehicle_type
                    }
                    violations.append(violation)
                    print(f"⚠️ Safety violation detected: {violation_type} - Vehicle {track_id}")
                
        return violations
    
    def save_violation_enhanced(self, violation, frame, video_id=None):
        """Enhanced violation saving with error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs("static/violations", exist_ok=True)
            
            # Extract license plate - CORRECT METHOD NAME
            license_plate = self.detect_license_plate(frame, violation['bbox'])
            
            # Calculate fine and severity
            fine_amount = self.calculate_fine_amount(
                violation['type'], 
                violation.get('speed')
            )
            severity = self.determine_severity(
                violation['type'], 
                violation.get('speed')
            )
            
            # Save violation image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"{violation['type']}_{timestamp}.jpg"
            image_path = f"static/violations/{image_filename}"
            
            try:
                cv2.imwrite(image_path, frame)
            except Exception as e:
                print(f"⚠️ Error saving image: {e}")
                image_path = None
            
            # Save to database
            try:
                database_path = os.path.abspath('traffic_violations.db')
                conn = sqlite3.connect(database_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO violations 
                    (timestamp, violation_type, vehicle_id, vehicle_license_plate, 
                     confidence, image_path, speed, location, severity, fine_amount, video_id, vehicle_type, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    violation['timestamp'],
                    violation['type'],
                    violation.get('track_id', -1),
                    license_plate,
                    violation['confidence'],
                    image_path,
                    violation.get('speed', 0),
                    'Camera_1',
                    severity,
                    fine_amount,
                    video_id,
                    violation.get('vehicle_type', 'unknown'),
                    violation.get('description', f"{violation['type']} detected")
                ))
                
                conn.commit()
                conn.close()
                
                print(f"✅ Violation saved: {violation['type']} - Vehicle {violation.get('track_id')} - {severity} severity")
                return image_path
                
            except Exception as e:
                print(f"❌ Error saving violation to database: {e}")
                return None
                
        except Exception as e:
            print(f"❌ Error in save_violation_enhanced: {e}")
            return None
    
    def process_frame(self, frame):
        """Enhanced frame processing with comprehensive violation detection"""
        frame_time = time.time()
        
        try:
            # Detect traffic light state
            self.detect_traffic_light_state(frame)
            
            # Run YOLO detection with tracking
            results = self.model.track(frame, persist=True, conf=self.confidence_threshold, 
                                     classes=[0, 1, 2, 3, 5, 7])  # person, bicycle, car, motorcycle, bus, truck
            
            if results[0].boxes is None:
                return frame, []
            
            # Parse detections with enhanced information
            detections = []
            current_frame_vehicles = set()
            
            for box in results[0].boxes:
                if box.id is not None:
                    class_id = int(box.cls[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Convert bbox format for processing
                    bbox_processed = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # [x, y, w, h]
                    
                    # Only process vehicles (not persons)
                    if class_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                        current_frame_vehicles.add(track_id)
                        self.unique_vehicles.add(track_id)
                        
                        # Try to detect license plate
                        license_plate = self.detect_license_plate(frame, bbox_processed)
                        
                        detection = {
                            'bbox': bbox_processed,
                            'confidence': confidence,
                            'class_id': class_id,
                            'track_id': track_id,
                            'vehicle_type': self.vehicle_classes.get(class_id, 'unknown'),
                            'license_plate': license_plate,
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)
                        
                        self.total_detections += 1
            
            # Check for various violations
            all_violations = []
            
            # Red light violations
            red_light_violations = self.check_red_light_violation(detections, frame)
            all_violations.extend(red_light_violations)
            
            # Speed violations
            speed_violations = self.check_speed_violation(detections, frame_time)
            all_violations.extend(speed_violations)
            
            # Wrong way driving
            wrong_way_violations = self.check_wrong_way_driving(detections)
            all_violations.extend(wrong_way_violations)
            
            # Safety violations
            safety_violations = self.check_safety_violations(detections, frame)
            all_violations.extend(safety_violations)
            
            # Save violations with enhanced data
            for violation in all_violations:
                # Add license plate info if available
                for detection in detections:
                    if detection['track_id'] == violation.get('track_id'):
                        violation['license_plate'] = detection.get('license_plate')
                        violation['vehicle_type'] = detection.get('vehicle_type')
                        break
                
                # Save violation using the enhanced method
                try:
                    self.save_violation_enhanced(violation, frame, getattr(self, 'video_id', None))
                except Exception as e:
                    print(f"❌ Error saving violation: {e}")
            
            # Create annotated frame with enhanced information
            annotated_frame = results[0].plot()
            
            # Draw additional information
            self.draw_enhanced_interface_elements(annotated_frame, all_violations, detections)
            
            return annotated_frame, all_violations
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return frame, []
    
    def draw_enhanced_interface_elements(self, frame, violations, detections):
        """Draw enhanced interface elements on frame"""
        try:
            # Draw stop line
            cv2.line(frame, (0, self.stop_line_y), 
                    (frame.shape[1], self.stop_line_y), (0, 255, 255), 2)
            cv2.putText(frame, "STOP LINE", (10, self.stop_line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw traffic light state
            light_color = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0)}
            cv2.circle(frame, (50, 50), 20, light_color.get(self.traffic_light_state, (128, 128, 128)), -1)
            cv2.putText(frame, f"LIGHT: {self.traffic_light_state.upper()}", (80, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw violation count
            cv2.putText(frame, f"VIOLATIONS: {len(violations)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw vehicle count
            cv2.putText(frame, f"VEHICLES: {len(detections)}", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw unique vehicles
            cv2.putText(frame, f"UNIQUE: {len(self.unique_vehicles)}", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"⚠️ Error drawing interface elements: {e}")
        
        return frame
    
    def cleanup(self):
        """Clean up database connection"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception as e:
            print(f"⚠️ Error closing database connection: {e}")