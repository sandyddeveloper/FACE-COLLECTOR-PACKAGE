import cv2
import os
import time
import logging
import logging.handlers
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import requests
import io
import argparse
from pathlib import Path
from typing import Optional, Tuple
from facenet_pytorch import MTCNN
import threading
import queue

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_STREAM_URL = "http://192.168.68.101:8080/video"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_API_URL = "http://local.localhost:8000/img_check"

# API Metadata Defaults
DEFAULT_CAMERA_ID = "1"
DEFAULT_DEVICE_ID = "inference id testing"
DEFAULT_DEVICE_NAME = "suresh"
DEFAULT_ORG_ID = "7"

PROCESS_INTERVAL = 0.2  # Seconds between processing frames
DETECTION_WIDTH = 640   # Resize width for detection speedup
MIN_FACE_SIZE = 60      # Minimum face width/height in pixels
BLUR_THRESHOLD = 120    # Laplacian variance threshold (Optimized for CCTV)
POSE_THRESHOLD = 0.4    # Nose-to-eye ratio for frontal pose check
CONFIDENCE_THRESHOLD = 0.95 # MTCNN detection probability
COOLDOWN_PERIOD = 2.0   # Seconds to wait before sending same person again

def setup_logging(log_file: str = "face_collector.log"):
    """Sets up logging with rotating file handler and prevents duplication."""
    logger = logging.getLogger()
    
    # Clear existing handlers to prevent duplication in debug/restarts
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Rotating File Handler (Max 5MB, keeping 3 backups)
    fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class FaceCollector:
    def __init__(self, stream_url: str, base_output_dir: str, api_config: dict):
        self.stream_url = stream_url
        self.base_dir = Path(base_output_dir)
        self.api_config = api_config
        self.session = requests.Session() 
        
        # Async Networking: Background queue and thread
        self.api_queue = queue.Queue(maxsize=50)
        self.worker_thread = threading.Thread(target=self._api_worker, daemon=True)
        self.worker_thread.start()
        
        # Deduplication state
        self.last_send_time = 0
        
        self.device = self._detect_device()
        logging.info(f"Using device: {self.device}")
        
        # MTCNN for detection & landmarks
        self.mtcnn = MTCNN(keep_all=True, device=self.device, select_largest=False)

    def _detect_device(self) -> str:
        """Detects the best available hardware accelerator."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            # Support for MacOS M1/M2/M3 chips
            return "mps"
        return "cpu"

    def process_stream(self):
        """Main loop to capture and process video stream."""
        logging.info(f"Connecting to stream: {self.stream_url}")
        cap = cv2.VideoCapture(self.stream_url)
        
        if not cap.isOpened():
            logging.error("Could not open video stream. Check URL and network.")
            return

        last_process_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Stream lost. Attempting reconnect in 5s...")
                    cap.release()
                    cv2.destroyAllWindows() # Clean up windows before reconnecting
                    time.sleep(5)
                    cap = cv2.VideoCapture(self.stream_url)
                    continue

                # Non-blocking visualization
                cv2.imshow("Face Collector Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("User requested exit.")
                    break

                # Process throttling
                if time.time() - last_process_time < PROCESS_INTERVAL:
                    continue
                    
                last_process_time = time.time()
                self._process_frame(frame)

        except KeyboardInterrupt:
            logging.info("Stopping due to KeyboardInterrupt...")
        except Exception as e:
            logging.error(f"Unexpected error in stream loop: {e}")
        finally:
            cap.release()
            self.session.close()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray):
        """Detects, aligns, and saves high-quality faces from a single frame."""
        h, w = frame.shape[:2]
        scale = DETECTION_WIDTH / w if w > DETECTION_WIDTH else 1.0
        
        if scale < 1.0:
            detect_frame = cv2.resize(frame, (DETECTION_WIDTH, int(h * scale)))
        else:
            detect_frame = frame

        # Convert to RGB/PIL
        pil_detect = Image.fromarray(cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB))
        pil_full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        try:
            boxes, probs, points = self.mtcnn.detect(pil_detect, landmarks=True)
        except Exception:
            return 

        if boxes is None or probs is None:
            return

        # Rescale boxes/points back to original resolution
        if scale < 1.0:
            boxes = boxes / scale
            if points is not None:
                points = points / scale

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < CONFIDENCE_THRESHOLD: continue
            
            try:
                # Coordinate extraction
                x1, y1, x2, y2 = [int(b) for b in box]
                face_w, face_h = x2 - x1, y2 - y1
                
                if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE: continue

                # Safe Crop Padding
                pad = int(face_w * 0.4) # 40% padding for better backend detection
                p_x1 = max(0, x1 - pad)
                p_y1 = max(0, y1 - pad)
                p_x2 = min(w, x2 + pad)
                p_y2 = min(h, y2 + pad)
                
                face_crop = pil_full.crop((p_x1, p_y1, p_x2, p_y2))
                
                if points is None: continue

                # Alignment & Filtering logic updated:
                # We no longer manually align here because the backend handles rotation
                # and manual rotation adds black borders/tight crops that fail detection.
                if self._is_good_quality(face_crop, points[i]):
                     # Final Blur Check on the crop directly
                     blur_score = self._get_blur_score(face_crop)
                     if blur_score >= BLUR_THRESHOLD:
                         self._send_to_api(face_crop, timestamp, i, blur_score)

                # Explicit memory cleanup
                del face_crop

            except Exception as e:
                logging.error(f"Error processing face {i}: {e}")
        
        # Explicit memory cleanup for frame objects
        del pil_detect
        del pil_full

    def _is_good_quality(self, face_crop: Image.Image, landmark: np.ndarray) -> bool:
        """Checks Pose (Frontal logic)."""
        left_eye, right_eye, nose = landmark[0], landmark[1], landmark[2]
        
        # Pose: Ratio of nose distance to eyes
        l_dist = np.linalg.norm(left_eye - nose)
        r_dist = np.linalg.norm(right_eye - nose)
        
        if max(l_dist, r_dist) == 0: return False
        if min(l_dist, r_dist) / max(l_dist, r_dist) < POSE_THRESHOLD:
            return False 
            
        return True

    def _get_blur_score(self, pil_img: Image.Image) -> float:
        """Calculates Laplacian Variance."""
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _send_to_api(self, pil_img: Image.Image, timestamp: str, idx: int, score: float):
        """Adds face to background processing queue."""
        # Simple cooldown deduplication
        current_time = time.time()
        if current_time - self.last_send_time < COOLDOWN_PERIOD:
            return
        
        self.last_send_time = current_time
        
        filename = f"Face_{timestamp}_{idx}_S{int(score)}.jpg"
        
        try:
            # Prepare image in memory
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Queue data for background thread
            data = {
                'filename': filename,
                'content': img_byte_arr.getvalue(),
                'payload': {
                    'camera_id': self.api_config['camera_id'],
                    'device_id': self.api_config['device_id'],
                    'device_name': self.api_config['device_name'],
                    'org_id': self.api_config['org_id']
                }
            }
            
            try:
                self.api_queue.put_nowait(data)
            except queue.Full:
                logging.warning("API Queue full, dropping frame.")
                
        except Exception as e:
            logging.error(f"Error preparing image for queue: {e}")

    def _api_worker(self):
        """Background thread to handle API requests without blocking video stream."""
        while True:
            try:
                item = self.api_queue.get()
                if item is None: break
                
                filename = item['filename']
                files = {'image': (filename, item['content'], 'image/jpeg')}
                
                try:
                    response = self.session.post(
                        self.api_config['api_url'], 
                        files=files, 
                        data=item['payload'], 
                        timeout=60
                    )
                    response.raise_for_status() 
                    logging.info(f"Successfully sent {filename} to API (Background thread).")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Network error sending {filename} in background: {e}")
                
                self.api_queue.task_done()
            except Exception as e:
                logging.error(f"Unexpected error in API worker thread: {e}")

def main():
    parser = argparse.ArgumentParser(description="Professional Face Collector CLI")
    parser.add_argument("--stream-url", type=str, default=DEFAULT_STREAM_URL, help="Video stream source")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--log-file", type=str, default="face_collector.log", help="Log file path")
    
    # API Metadata
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Backend API endpoint")
    parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--device-id", type=str, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--device-name", type=str, default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--org-id", type=str, default=DEFAULT_ORG_ID)
    
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    api_config = {
        'api_url': args.api_url,
        'camera_id': args.camera_id,
        'device_id': args.device_id,
        'device_name': args.device_name,
        'org_id': args.org_id
    }
    
    try:
        collector = FaceCollector(args.stream_url, args.output_dir, api_config)
        collector.process_stream()
    except Exception as e:
        logging.critical(f"Fatal error in application: {e}")

if __name__ == "__main__":
    main()
