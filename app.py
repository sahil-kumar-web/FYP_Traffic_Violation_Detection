# app.py - FINAL VERSION - Complete Traffic Violation Detection API
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sqlite3
import json
from datetime import datetime, timedelta
import os
import uuid
from pathlib import Path
import cv2
import asyncio
from typing import List, Dict, Any
import shutil

# Import our enhanced YOLO traffic violation detector
from traffic_detector import TrafficViolationDetector

app = FastAPI(title="Traffic Violation Detection API - Enhanced Version")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/violations", exist_ok=True)
os.makedirs("../uploads", exist_ok=True)  # Create uploads in main project folder

# Static files for violation images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE_PATH = "traffic_violations.db"

def init_enhanced_database():
    """Initialize enhanced database schema for the dashboard"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Enhanced violations table with all necessary fields
    cursor.execute('''
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
    
    # Enhanced videos table for tracking uploaded videos
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original_name TEXT,
            file_path TEXT,
            file_size INTEGER,
            status TEXT DEFAULT 'pending',
            upload_time TEXT DEFAULT CURRENT_TIMESTAMP,
            processed_time TEXT,
            total_violations INTEGER DEFAULT 0,
            total_vehicles INTEGER DEFAULT 0,
            processing_duration REAL,
            error_message TEXT
        )
    ''')
    
    # Vehicle tracking table for analytics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT,
            vehicle_type TEXT,
            first_seen TEXT,
            last_seen TEXT,
            total_violations INTEGER DEFAULT 0,
            video_id INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_enhanced_database()

# Global enhanced detector instance
detector = TrafficViolationDetector()

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Traffic Violation Detection API - Enhanced Version",
        "version": "2.0",
        "features": [
            "Enhanced vehicle detection and counting",
            "License plate recognition with OCR",
            "Comprehensive violation tracking",
            "Real-time dashboard data",
            "Video upload and processing",
            "Detailed analytics and statistics"
        ],
        "endpoints": {
            "dashboard": "/api/dashboard/",
            "violations": "/api/violations/all",
            "statistics": "/api/statistics/detailed",
            "upload": "/api/upload-video/",
            "docs": "/docs"
        }
    }

@app.get("/api/dashboard/")
async def get_dashboard_data():
    """Get comprehensive dashboard statistics and violations"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get total violations
        cursor.execute("SELECT COUNT(*) FROM violations")
        total_violations = cursor.fetchone()[0]
        
        # Get total unique vehicles (both from license plates and tracking IDs)
        cursor.execute("""
            SELECT COUNT(DISTINCT CASE 
                WHEN vehicle_license_plate IS NOT NULL AND vehicle_license_plate != '' 
                THEN vehicle_license_plate 
                ELSE 'VEH_' || vehicle_id 
            END) FROM violations
        """)
        total_vehicles = cursor.fetchone()[0]
        
        # Get total videos
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_videos = cursor.fetchone()[0]
        
        # Get pending videos
        cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'pending' OR status = 'processing'")
        pending_videos = cursor.fetchone()[0]
        
        # Get recent violations (last 20 for UI, but also provide all)
        cursor.execute('''
            SELECT id, timestamp, violation_type, vehicle_license_plate, location, 
                   severity, fine_amount, is_resolved, speed, vehicle_id, confidence,
                   image_path, vehicle_type, description
            FROM violations 
            ORDER BY timestamp DESC 
            LIMIT 20
        ''')
        recent_violations = []
        for row in cursor.fetchall():
            recent_violations.append({
                "id": row[0],
                "timestamp": row[1],
                "violation_type": row[2],
                "vehicle_license_plate": row[3] or f"VEH-{row[9]}" if row[9] else "Unknown",
                "location": row[4],
                "severity": row[5],
                "fine_amount": row[6],
                "is_resolved": bool(row[7]),
                "speed": row[8],
                "vehicle_id": row[9],
                "confidence": round(row[10], 3) if row[10] else 0,
                "image_path": row[11],
                "vehicle_type": row[12] or "Unknown",
                "description": row[13] or ""
            })
        
        # Get violation trends for last 30 days
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count, 
                   GROUP_CONCAT(violation_type) as types,
                   SUM(fine_amount) as daily_fines
            FROM violations 
            WHERE timestamp >= DATE('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''')
        violation_trends = []
        for row in cursor.fetchall():
            violation_trends.append({
                "date": row[0], 
                "count": row[1],
                "types": row[2].split(',') if row[2] else [],
                "fines": row[3] if row[3] else 0
            })
        
        # Get detailed statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_severity,
                SUM(CASE WHEN severity = 'medium' THEN 1 ELSE 0 END) as medium_severity,
                SUM(CASE WHEN severity = 'low' THEN 1 ELSE 0 END) as low_severity,
                SUM(fine_amount) as total_fines,
                COUNT(DISTINCT vehicle_license_plate) as unique_plates,
                AVG(confidence) as avg_confidence
            FROM violations
        ''')
        stats = cursor.fetchone()
        
        # Get processing statistics
        cursor.execute('''
            SELECT 
                status,
                COUNT(*) as count,
                SUM(total_violations) as violations_found,
                SUM(total_vehicles) as vehicles_detected,
                AVG(processing_duration) as avg_processing_time
            FROM videos
            GROUP BY status
        ''')
        video_stats = {}
        for row in cursor.fetchall():
            video_stats[row[0]] = {
                "count": row[1], 
                "violations": row[2] or 0, 
                "vehicles": row[3] or 0,
                "avg_processing_time": round(row[4], 2) if row[4] else 0
            }
        
        conn.close()
        
        return {
            "total_violations": total_violations,
            "total_vehicles": total_vehicles,
            "total_videos": total_videos,
            "pending_videos": pending_videos,
            "recent_violations": recent_violations,
            "violation_trends": violation_trends,
            "detailed_stats": {
                "total_violations": stats[0] if stats[0] else 0,
                "high_severity": stats[1] if stats[1] else 0,
                "medium_severity": stats[2] if stats[2] else 0,
                "low_severity": stats[3] if stats[3] else 0,
                "total_fines": round(stats[4], 2) if stats[4] else 0,
                "unique_license_plates": stats[5] if stats[5] else 0,
                "avg_confidence": round(stats[6], 3) if stats[6] else 0
            },
            "video_processing_stats": video_stats,
            "system_status": "active",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/violation-summary/")
async def get_violation_summary():
    """Get violation types summary for charts"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT violation_type, COUNT(*) as count, 
                   AVG(confidence) as avg_confidence,
                   SUM(fine_amount) as total_fines
            FROM violations 
            GROUP BY violation_type
            ORDER BY count DESC
        ''')
        
        summary = []
        for row in cursor.fetchall():
            summary.append({
                "violation_type": row[0], 
                "count": row[1],
                "avg_confidence": round(row[2], 3) if row[2] else 0,
                "total_fines": round(row[3], 2) if row[3] else 0
            })
        
        conn.close()
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/violations/all")
async def get_all_violations():
    """Get all violations with complete details"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, violation_type, vehicle_id, vehicle_license_plate,
                   confidence, image_path, speed, location, severity, fine_amount,
                   is_resolved, created_at, vehicle_type, description
            FROM violations 
            ORDER BY timestamp DESC
        ''')
        
        violations = []
        for row in cursor.fetchall():
            violations.append({
                "id": row[0],
                "timestamp": row[1],
                "violation_type": row[2],
                "vehicle_id": row[3],
                "vehicle_license_plate": row[4] or f"VEH-{row[3]}",
                "confidence": round(row[5], 3) if row[5] else 0,
                "image_path": row[6],
                "speed": row[7],
                "location": row[8],
                "severity": row[9],
                "fine_amount": row[10],
                "is_resolved": bool(row[11]),
                "created_at": row[12],
                "vehicle_type": row[13] or "Unknown",
                "description": row[14] or ""
            })
        
        conn.close()
        return {"violations": violations, "total": len(violations)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/statistics/detailed")
async def get_detailed_statistics():
    """Get comprehensive system statistics"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Violation type breakdown
        cursor.execute('''
            SELECT violation_type, COUNT(*) as count, 
                   AVG(confidence) as avg_confidence,
                   SUM(fine_amount) as total_fines,
                   COUNT(DISTINCT vehicle_license_plate) as unique_vehicles
            FROM violations 
            GROUP BY violation_type
            ORDER BY count DESC
        ''')
        violation_breakdown = []
        for row in cursor.fetchall():
            violation_breakdown.append({
                "type": row[0],
                "count": row[1], 
                "avg_confidence": round(row[2], 3) if row[2] else 0,
                "total_fines": round(row[3], 2) if row[3] else 0,
                "unique_vehicles": row[4] if row[4] else 0
            })
        
        # Daily violation trends
        cursor.execute('''
            SELECT DATE(timestamp) as date, 
                   COUNT(*) as total_violations,
                   COUNT(DISTINCT vehicle_id) as unique_vehicles,
                   SUM(fine_amount) as daily_fines,
                   AVG(confidence) as avg_confidence
            FROM violations 
            WHERE timestamp >= DATE('now', '-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''')
        daily_trends = []
        for row in cursor.fetchall():
            daily_trends.append({
                "date": row[0],
                "violations": row[1],
                "unique_vehicles": row[2],
                "fines": round(row[3], 2) if row[3] else 0,
                "avg_confidence": round(row[4], 3) if row[4] else 0
            })
        
        # Vehicle repeat offenders
        cursor.execute('''
            SELECT vehicle_license_plate, COUNT(*) as violation_count,
                   GROUP_CONCAT(DISTINCT violation_type) as violation_types,
                   SUM(fine_amount) as total_fines,
                   MAX(timestamp) as last_violation
            FROM violations 
            WHERE vehicle_license_plate IS NOT NULL AND vehicle_license_plate != ''
            GROUP BY vehicle_license_plate
            HAVING violation_count > 1
            ORDER BY violation_count DESC
            LIMIT 20
        ''')
        repeat_offenders = []
        for row in cursor.fetchall():
            repeat_offenders.append({
                "license_plate": row[0],
                "violations": row[1],
                "violation_types": row[2].split(',') if row[2] else [],
                "total_fines": round(row[3], 2) if row[3] else 0,
                "last_violation": row[4]
            })
        
        conn.close()
        
        return {
            "violation_breakdown": violation_breakdown,
            "daily_trends": daily_trends,
            "repeat_offenders": repeat_offenders,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """Upload video for processing with enhanced tracking"""
    try:
        # Validate file type
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        
        # Create uploads directory in the main project folder
        uploads_dir = os.path.abspath("../uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(video.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(uploads_dir, unique_filename)
        
        # Save uploaded file
        content = await video.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        file_size = len(content)
        
        print(f"✅ Video uploaded successfully!")
        print(f"📁 Original name: {video.filename}")
        print(f"💾 Saved as: {unique_filename}")
        print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
        print(f"🗂️ Path: {file_path}")
        
        # Save video record to database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO videos (filename, original_name, file_path, file_size, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (unique_filename, video.filename, file_path, file_size, 'pending'))
        
        video_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Add background task to process video
        background_tasks.add_task(process_video_background, video_id, file_path)
        
        return {
            "message": "Video uploaded successfully",
            "video_id": video_id,
            "filename": unique_filename,
            "original_name": video.filename,
            "file_path": file_path,
            "file_size_mb": round(file_size / (1024*1024), 2),
            "status": "pending"
        }
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

async def process_video_background(video_id: int, file_path: str):
    """Enhanced background task to process uploaded video"""
    start_time = datetime.now()
    
    try:
        print(f"🎬 Starting video processing for ID: {video_id}")
        print(f"📁 File path: {file_path}")
        
        # Update video status to processing
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE videos SET status = 'processing' WHERE id = ?", (video_id,))
        conn.commit()
        conn.close()
        
        # Reset detector for this video
        detector.video_id = video_id
        detector.unique_vehicles = set()
        detector.total_detections = 0
        detector.processed_violations = set()
        
        # Process video
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {file_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        total_violations = 0
        frame_count = 0
        skip_frames = 3  # Process every 3rd frame for better performance
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % skip_frames != 0:
                continue
            
            try:
                # Process frame for violations
                processed_frame, violations = detector.process_frame(frame)
                total_violations += len(violations)
                
                # Progress logging every 10%
                if frame_count % (max(1, total_frames // 10)) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ Progress: {progress:.1f}% - Violations: {total_violations}, Vehicles: {len(detector.unique_vehicles)}")
                    
            except Exception as e:
                print(f"⚠️  Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        
        # Calculate processing time
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        # Final statistics
        unique_vehicle_count = len(detector.unique_vehicles)
        
        print(f"✅ Video processing completed!")
        print(f"📊 Final stats: {total_violations} violations, {unique_vehicle_count} unique vehicles")
        print(f"⏱️ Processing time: {processing_duration:.2f} seconds")
        
        # Update video record with comprehensive results
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE videos 
            SET status = 'completed', 
                processed_time = ?, 
                total_violations = ?, 
                total_vehicles = ?,
                processing_duration = ?
            WHERE id = ?
        ''', (end_time.isoformat(), total_violations, unique_vehicle_count, processing_duration, video_id))
        conn.commit()
        conn.close()
        
        print(f"🎯 Video {video_id} processing complete!")
        
    except Exception as e:
        # Calculate processing time even for errors
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        print(f"❌ Error processing video {video_id}: {str(e)}")
        
        # Update video status to error with error message
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE videos 
            SET status = 'error', 
                processed_time = ?,
                processing_duration = ?,
                error_message = ?
            WHERE id = ?
        ''', (end_time.isoformat(), processing_duration, str(e), video_id))
        conn.commit()
        conn.close()

@app.get("/api/videos/")
async def get_videos():
    """Get list of uploaded videos with detailed information"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, original_name, filename, status, upload_time, processed_time, 
                   total_violations, total_vehicles, file_size, processing_duration,
                   error_message
            FROM videos 
            ORDER BY upload_time DESC
        ''')
        
        videos = []
        for row in cursor.fetchall():
            videos.append({
                "id": row[0],
                "original_name": row[1],
                "filename": row[2],
                "status": row[3],
                "upload_time": row[4],
                "processed_time": row[5],
                "total_violations": row[6],
                "total_vehicles": row[7],
                "file_size_mb": round(row[8] / (1024*1024), 2) if row[8] else 0,
                "processing_duration": round(row[9], 2) if row[9] else 0,
                "error_message": row[10]
            })
        
        conn.close()
        return videos
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/system/status")
async def get_system_status():
    """Get current system status and health"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get system statistics
        cursor.execute("SELECT COUNT(*) FROM violations")
        total_violations = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_videos = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM videos WHERE status = 'processing'")
        processing_videos = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "online",
            "version": "2.0",
            "database_connected": True,
            "yolo_model_loaded": detector.model is not None,
            "ocr_available": getattr(detector, 'ocr_available', False),
            "total_violations": total_violations,
            "total_videos": total_videos,
            "processing_videos": processing_videos,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    print("🚗 Traffic Violation Detection System - Enhanced Version")
    print("=" * 60)
    print("🔧 Features enabled:")
    print("   ✅ Enhanced vehicle detection and counting")
    print("   ✅ License plate recognition with OCR")
    print("   ✅ Comprehensive violation tracking")
    print("   ✅ Real-time dashboard data")
    print("   ✅ Video upload to uploads folder")
    print("   ✅ Detailed analytics and statistics")
    print("=" * 60)
    print("🌐 Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)