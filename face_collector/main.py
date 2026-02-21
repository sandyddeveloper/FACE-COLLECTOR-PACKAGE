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

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_STREAM_URL = "http://192.168.68.103:8080/video"
DEFAULT_OUTPUT_DIR = "output"
API_URL = "http://127.0.0.1:8000/img_check"
API_HOSTNAME = "local.localhost"

PROCESS_INTERVAL = 0.2  # Seconds between processing frames
DETECTION_WIDTH = 640   # Resize width for detection speedup
MIN_FACE_SIZE = 60      # Minimum face width/height in pixels
BLUR_THRESHOLD = 80     # Laplancian variance threshold
POSE_THRESHOLD = 0.4    # Nose-to-eye ratio for frontal pose check
CONFIDENCE_THRESHOLD = 0.95 # MTCNN detection probability

def setup_logging(log_file: str = "face_collector.log"):
    """Sets up logging with rotating file handler."""
    logger = logging.getLogger()
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
    def __init__(self, stream_url: str, base_output_dir: str):
        self.stream_url = stream_url
        self.base_dir = Path(base_output_dir)
        self.faces_dir = self.base_dir / "faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
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

        if boxes is None:
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
                pad = int(face_w * 0.1) # 10% padding
                p_x1 = max(0, x1 - pad)
                p_y1 = max(0, y1 - pad)
                p_x2 = min(w, x2 + pad)
                p_y2 = min(h, y2 + pad)
                
                face_crop = pil_full.crop((p_x1, p_y1, p_x2, p_y2))
                
                if points is None: continue

                # Alignment & Filtering
                if self._is_good_quality(face_crop, points[i]):
                     aligned_face = self._align_face(face_crop, points[i])
                     
                     # Final Blur Check on ALIGNED face
                     blur_score = self._get_blur_score(aligned_face)
                     if blur_score >= BLUR_THRESHOLD:
                         self._save_face(aligned_face, timestamp, i, blur_score)

            except Exception as e:
                logging.error(f"Error processing face {i}: {e}")

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

    def _align_face(self, face_crop: Image.Image, landmark: np.ndarray) -> Image.Image:
        """Rotates face to make eyes horizontal."""
        left_eye, right_eye = landmark[0], landmark[1]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return face_crop.rotate(angle, expand=True)

    def _get_blur_score(self, pil_img: Image.Image) -> float:
        """Calculates Laplacian Variance."""
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _save_face(self, pil_img: Image.Image, timestamp: str, idx: int, score: float):
        filename = f"Face_{timestamp}_{idx}_S{int(score)}.jpg"
        path = self.faces_dir / filename
        
        try:
            pil_img.save(path)
            logging.info(f"Saved: {filename}")
        except Exception as e:
            logging.error(f"Failed to save face to disk: {e}")
            return

        # Send image to API
        try:
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            files = {'image': (filename, img_byte_arr, 'image/jpeg')}
            
            headers = {"Host": API_HOSTNAME}
            response = requests.post(API_URL, files=files, headers=headers, timeout=10)
            response.raise_for_status() # Raise error for 4xx/5xx responses
            logging.info(f"Successfully sent {filename} to API.")
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error sending {filename} to API: {e}")
        except Exception as e:
            logging.error(f"Unexpected error sending {filename} to API: {e}")

def main():
    parser = argparse.ArgumentParser(description="Professional Face Collector CLI")
    parser.add_argument("--stream-url", type=str, default=DEFAULT_STREAM_URL, help="Video stream source")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--log-file", type=str, default="face_collector.log", help="Log file path")
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    try:
        collector = FaceCollector(args.stream_url, args.output_dir)
        collector.process_stream()
    except Exception as e:
        logging.critical(f"Fatal error in application: {e}")

if __name__ == "__main__":
    main()
