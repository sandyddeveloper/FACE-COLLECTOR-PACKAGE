import cv2
import os
import time
import logging
import logging.handlers
import numpy as np
from datetime import datetime
from PIL import Image
import threading
import queue
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
import requests
import io
import argparse
from pathlib import Path
from typing import Optional, Tuple
from facenet_pytorch import MTCNN

# CONFIGURATION
DEFAULT_STREAM_URL = "http://192.168.0.6:8080/video"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_API_URL = "https://uatbase.faceviz.com/img_check"


# API Metadata Defaults
DEFAULT_CAMERA_ID = "0"
DEFAULT_DEVICE_ID = "anbu"
DEFAULT_DEVICE_NAME = "anbu"
DEFAULT_ORG_ID = "3"

PROCESS_INTERVAL = 0.2  # Seconds between processing frames
DETECTION_WIDTH = 640   # Resize width for detection speedup
MIN_FACE_SIZE = 60      # Minimum face width/height in pixels
BLUR_THRESHOLD = 80     # Laplancian variance threshold
POSE_THRESHOLD = 0.4    # Nose-to-eye ratio for frontal pose check
CONFIDENCE_THRESHOLD = 0.95 # MTCNN detection probability
COOLDOWN_PERIOD = 300.0 # 5 minutes before sending the same grid location again

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

class VideoStream:
    """Bufferless Video Capture to ensure real-time frame processing."""
    def __init__(self, src: str):
        self.src = src
        self.cap = None
        self.thread = None
        self.ret, self.frame = False, None
        self.stopped = False
        self.lock = threading.Lock()
        
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            logging.error(f"VideoStream failed to open source: {src}")
            self.stopped = True
            return

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            if not ret:
                logging.warning("VideoStream lost connection. Internal thread stopping.")
                self.stopped = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

class FaceCollector:
    def __init__(self, stream_url: str, base_output_dir: str, api_config: dict, schedule_config: dict = None):
        self.stream_url = stream_url
        self.base_dir = Path(base_output_dir)
        self.faces_dir = self.base_dir / "faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_config = api_config
        self.schedule_config = schedule_config or {}
        
        # API Resilience: Retry strategy with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=1, # Wait 1s, 2s, 4s...
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Dynamic Blur Scaling state
        self.frame_gray_var = BLUR_THRESHOLD  # Initialize with static threshold
        self.last_blur_update = 0
        
        # Grid Tracking Cooldown (16x16 grid for better precision)
        self.grid_cooldowns = {} 
        
        # Grid Resolution
        self.GRID_SIZE = 16
        
        # Async Networking: Background queue and thread
        self.api_queue = queue.Queue(maxsize=100)
        self.running = True
        self.worker_thread = threading.Thread(target=self._api_worker, daemon=True)
        self.worker_thread.start()
        
        self.vs = None
        
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

    def _is_within_schedule(self) -> bool:
        """
        Checks if the current time matches the configured schedule (days of week, start/end time).
        """
        if not self.schedule_config:
            return True
        
        now = datetime.now()
        
        # Check active days
        run_days = self.schedule_config.get('run_days')
        if run_days:
            current_day = now.strftime('%A')[:3].lower()  # e.g., 'mon', 'tue'
            if current_day not in run_days:
                return False
                
        # Check time range
        start_time = self.schedule_config.get('start_time')
        end_time = self.schedule_config.get('end_time')
        current_time_str = now.strftime("%H:%M")
        
        if start_time and end_time:
            if start_time <= end_time:
                if not (start_time <= current_time_str <= end_time):
                    return False
            else:
                # Crosses midnight (e.g., 22:00 to 06:00)
                if not (start_time <= current_time_str or current_time_str <= end_time):
                    return False
        elif start_time and current_time_str < start_time:
            return False
        elif end_time and current_time_str > end_time:
            return False
            
        return True

    def process_stream(self):
        """Main loop to capture and process video stream."""
        logging.info(f"Connecting to stream: {self.stream_url}")
        self.vs = None
        
        last_process_time = 0
        schedule_logged = False
        
        try:
            while self.running:
                if not self._is_within_schedule():
                    if not schedule_logged:
                        logging.info("Outside of scheduled run hours/days. Pausing processing...")
                        schedule_logged = True
                    if self.vs:
                        self.vs.stop()
                        self.vs = None
                    time.sleep(10)
                    continue
                else:
                    if schedule_logged:
                        logging.info("Entering scheduled run hours. Resuming processing...")
                        schedule_logged = False

                # Handle VideoStream initialization or failure
                if self.vs is None or self.vs.stopped:
                    if self.vs: self.vs.stop()
                    logging.info("Connecting to VideoStream...")
                    self.vs = VideoStream(self.stream_url)
                    if self.vs.stopped:
                        logging.warning("Stream connection failed. Retrying in 5s...")
                        time.sleep(5)
                        continue

                ret, frame = self.vs.read()
                if not ret or frame is None:
                    logging.warning("Frame read failed. Checking stream...")
                    time.sleep(1) # Small gap before retry
                    continue

                # Non-blocking visualization
                cv2.imshow("Face Collector Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("User requested exit.")
                    self.stop()
                    break

                # Process throttling
                if time.time() - last_process_time < PROCESS_INTERVAL:
                    continue
                    
                last_process_time = time.time()
                
                # Dynamic Blur Scaling: Update scene variance every 5 seconds
                if time.time() - self.last_blur_update > 5.0:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.frame_gray_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
                    self.last_blur_update = time.time()
                
                self._process_frame(frame)

        except KeyboardInterrupt:
            logging.info("Stopping due to KeyboardInterrupt...")
            self.stop()
        except Exception as e:
            logging.error(f"Unexpected error in stream loop: {e}")
            self.stop()
        finally:
            self._cleanup()

    def _cleanup(self):
        """Final cleanup and queue draining log."""
        if self.vs:
            self.vs.stop()
        
        remaining = self.api_queue.qsize()
        if remaining > 0:
            logging.info(f"Shutdown: Draining {remaining} remaining items from queue...")
            # Wait a bit for the worker to finish or just log the loss
            time.sleep(2)
        
        self.session.close()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete.")

    def stop(self):
        """Sets the running flag to False."""
        self.running = False

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

                # Unique Faces: Grid Tracking Cooldown
                # Divide the frame into an 16x16 grid for higher precision
                grid_x = int((x1 + x2) / 2 / w * self.GRID_SIZE)
                grid_y = int((y1 + y2) / 2 / h * self.GRID_SIZE)
                grid_id = f"{grid_x}_{grid_y}"
                
                current_time = time.time()
                if grid_id in self.grid_cooldowns:
                    if current_time - self.grid_cooldowns[grid_id] < COOLDOWN_PERIOD:
                        continue # Skip if this spot was recently processed
                
                self.grid_cooldowns[grid_id] = current_time

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
                     # Final Blur & Exposure Checks
                     blur_score = self._get_blur_score(face_crop)
                     if self._is_sharp_enough(face_crop, self.frame_gray_var) and self._is_well_exposed(face_crop):
                          self._send_to_api(face_crop, timestamp, i, blur_score)

            except Exception as e:
                logging.error(f"Error processing face {i}: {e}")

    def _is_good_quality(self, face_crop: Image.Image, landmark: np.ndarray) -> bool:
        """Checks Pose and Head-Tilt Compensation."""
        left_eye, right_eye, nose = landmark[0], landmark[1], landmark[2]
        left_mouth, right_mouth = landmark[3], landmark[4]
        
        # 1. Pose: Horizontal Symmetry (Frontal logic)
        l_dist = np.linalg.norm(left_eye - nose)
        r_dist = np.linalg.norm(right_eye - nose)
        
        if max(l_dist, r_dist) == 0: return False
        if min(l_dist, r_dist) / max(l_dist, r_dist) < POSE_THRESHOLD:
            return False 
            
        # 2. Head-Tilt Compensation: Vertical Ratio
        eye_midpoint = (left_eye + right_eye) / 2
        mouth_midpoint = (left_mouth + right_mouth) / 2
        
        # Calculate vertical eye-to-nose vs nose-to-mouth ratio
        upper_dist = np.linalg.norm(eye_midpoint[1] - nose[1])
        lower_dist = np.linalg.norm(nose[1] - mouth_midpoint[1])
        
        # If the nose is almost touching the mouth (ratio < 0.2), they are looking too far down
        if lower_dist / (upper_dist + 1e-6) < 0.2:
            return False
            
        return True

    def _is_sharp_enough(self, face_crop, frame_gray_var):
        """Dynamic Blur Scaling: Compare face blur relative to whole frame."""
        face_blur = self._get_blur_score(face_crop)
        # If the face is significantly blurrier than the background, discard it
        if face_blur < (frame_gray_var * 0.8): 
            return False
        return face_blur > BLUR_THRESHOLD

    def _is_well_exposed(self, pil_img):
        """CIE Lab Brightness Filter: Check for underexposed or overexposed faces."""
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2LAB)
        l_channel, _, _ = cv2.split(cv_img)
        avg_brightness = np.mean(l_channel)
        # 50 is too dark, 240 is too bright/glare
        return 50 < avg_brightness < 240 

    def _get_blur_score(self, pil_img: Image.Image) -> float:
        """Calculates Laplacian Variance."""
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _send_to_api(self, pil_img: Image.Image, timestamp: str, idx: int, score: float):
        """Saves face locally and adds to background processing queue."""
        filename = f"Face_{timestamp}_{idx}_S{int(score)}.jpg"
        path = self.faces_dir / filename
        
        try:
            # Save locally
            pil_img.save(path)
            logging.info(f"Saved locally: {filename}")
            
            # Prepare image in memory for API
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
                logging.info(f"Queued: {filename}")
            except queue.Full:
                logging.warning("API Queue full, dropping frame.")
                
        except Exception as e:
            logging.error(f"Error preparing image for queue: {e}")

    def _api_worker(self):
        """Background thread to handle API requests without blocking video stream."""
        while self.running or not self.api_queue.empty():
            try:
                try:
                    item = self.api_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
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
                    error_msg = f"Network error sending {filename} in background: {e}"
                    if hasattr(e, 'response') and e.response is not None:
                        error_msg += f" - Response: {e.response.text}"
                    logging.error(error_msg)
                
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
    
    # Automation / Schedule Metadata
    parser.add_argument("--start-time", type=str, default=None, help="Start time in HH:MM format (e.g. 09:00)")
    parser.add_argument("--end-time", type=str, default=None, help="End time in HH:MM format (e.g. 18:00)")
    parser.add_argument("--run-days", type=str, default=None, help="Days to run (e.g., 'Mon,Tue', 'Monday to Friday', 'All')")
    
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    api_config = {
        'api_url': args.api_url,
        'camera_id': args.camera_id,
        'device_id': args.device_id,
        'device_name': args.device_name,
        'org_id': args.org_id
    }
    
    schedule_config = {}
    if args.start_time:
        schedule_config['start_time'] = args.start_time.strip()
    if args.end_time:
        schedule_config['end_time'] = args.end_time.strip()
    if args.run_days:
        raw_days = args.run_days.lower()
        if 'all' not in raw_days:
            all_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
            active_days = set()
            parts = raw_days.replace(' to ', '-').split(',')
            for part in parts:
                if '-' in part:
                    s_str, e_str = part.split('-')[:2]
                    s, e = s_str.strip()[:3], e_str.strip()[:3]
                    if s in all_days and e in all_days:
                        idx1, idx2 = all_days.index(s), all_days.index(e)
                        if idx1 <= idx2:
                            active_days.update(all_days[idx1:idx2+1])
                        else:
                            active_days.update(all_days[idx1:] + all_days[:idx2+1])
                else:
                    d = part.strip()[:3]
                    if d in all_days:
                        active_days.add(d)
            if active_days:
                schedule_config['run_days'] = list(active_days)
    
    try:
        collector = FaceCollector(args.stream_url, args.output_dir, api_config, schedule_config)
        collector.process_stream()
    except Exception as e:
        logging.critical(f"Fatal error in application: {e}")

if __name__ == "__main__":
    main()
