import os
# Silence FFMPEG warnings (overread, etc.)
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["FFMPEG_LOG_LEVEL"] = "quiet"

import cv2
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
import argparse
from pathlib import Path
from typing import Optional, Tuple
from facenet_pytorch import MTCNN
 


# CONFIGURATION
DEFAULT_STREAM_URL = "http://192.168.1.103:8080/video"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_API_URL = "https://uatbase.faceviz.com/img_check"


# API Metadata Defaults
DEFAULT_CAMERA_ID = "0"
DEFAULT_DEVICE_ID = "checking01"
DEFAULT_DEVICE_NAME = "checking"
DEFAULT_ORG_ID = "33"

PROCESS_INTERVAL = 0.01  
DETECTION_WIDTH = 480   
MIN_FACE_SIZE = 120      # Stricter for high accuracy (target > 160 for best)
BLUR_THRESHOLD = 100    # Stricter for dlib
DARKNESS_THRESHOLD = 80 
POSE_THRESHOLD = 0.85   # Strictly frontal faces only
CONFIDENCE_THRESHOLD = 0.95 
TRACKING_MAX_DISTANCE = 100 
TRACKING_MAX_MISSING = 5    
MAX_TRACKING_FRAMES = 60    
API_JPEG_QUALITY = 95       # Ultra high quality

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
        
        # Tracking & Golden Frame State
        self.trackers = {} # track_id -> dict(last_box, best_quality, best_frame, best_landmark, frame_count, missing_count, timestamp)
        self.next_track_id = 0
        
        # Async Networking: Background queue and thread
        self.api_queue = queue.Queue(maxsize=100)
        self.running = True
        self.worker_thread = threading.Thread(target=self._api_worker, daemon=True)
        self.worker_thread.start()
        
        self.vs = None
        
        self.device = self._detect_device()
        logging.info(f"Using device: {self.device}")
        
        # MTCNN for detection & landmarks - optimized for speed
        self.mtcnn = MTCNN(
            keep_all=True, 
            device=self.device, 
            select_largest=False,
            min_face_size=20, 
            post_process=False        
        )

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
                    if self.vs:
                        self.vs.stop()
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
        show_footer()
        logging.info("Shutdown complete.")

    def stop(self):
        """Sets the running flag to False."""
        self.running = False

    def _process_frame(self, frame: np.ndarray):
        """Detects, tracks, and selects the best frames."""
        h, w = frame.shape[:2]
        scale = DETECTION_WIDTH / w if w > DETECTION_WIDTH else 1.0
        
        detect_frame = cv2.resize(frame, (DETECTION_WIDTH, int(h * scale))) if scale < 1.0 else frame
        pil_detect = Image.fromarray(cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB))
        
        try:
            boxes, probs, points = self.mtcnn.detect(pil_detect, landmarks=True)
        except Exception:
            return 

        current_boxes = []
        if boxes is not None and len(boxes) > 0:
            if scale < 1.0:
                boxes = boxes / scale
                if points is not None:
                    points = points / scale
            
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < CONFIDENCE_THRESHOLD: continue
                current_boxes.append({'box': box, 'landmark': points[i], 'prob': prob})

        self._update_tracks(frame, current_boxes)

    def _update_tracks(self, frame: np.ndarray, current_faces: list):
        """Updates tracking IDs and evaluates quality for each person in frame."""
        h, w = frame.shape[:2]
        matched_face_indices = set()
        
        # 1. Try to match existing trackers to new detections
        for track_id, info in list(self.trackers.items()):
            best_match_idx = -1
            min_dist = TRACKING_MAX_DISTANCE
            
            centroid_prev = self._get_centroid(info['last_box'])
            
            for i, face in enumerate(current_faces):
                if i in matched_face_indices: continue
                centroid_curr = self._get_centroid(face['box'])
                dist = np.linalg.norm(centroid_prev - centroid_curr)
                
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                matched_face_indices.add(best_match_idx)
                face = current_faces[best_match_idx]
                self._update_tracker_quality(track_id, frame, face)
                info['missing_count'] = 0
                info['last_box'] = face['box']
            else:
                info['missing_count'] += 1
                if info['missing_count'] > TRACKING_MAX_MISSING:
                    self._finalize_track(track_id)

        # 2. Spawn new trackers for unmatched detections
        for i, face in enumerate(current_faces):
            if i not in matched_face_indices:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.trackers[track_id] = {
                    'last_box': face['box'],
                    'best_quality': -1,
                    'best_frame': None,
                    'best_landmark': None,
                    'frame_count': 0,
                    'missing_count': 0,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'uploaded': False
                }
                self._update_tracker_quality(track_id, frame, face)

    def _get_centroid(self, box):
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    def _update_tracker_quality(self, track_id, frame, face):
        """Evaluates and updates the 'Golden Frame' for a track."""
        info = self.trackers[track_id]
        if info['uploaded']: return
        
        box = face['box']
        x1, y1, x2, y2 = [int(b) for b in box]
        face_w, face_h = x2 - x1, y2 - y1
        
        if face_w < MIN_FACE_SIZE: return

        # Crop with padding
        pad = int(face_w * 0.4)
        p_x1, p_y1 = max(0, x1 - pad), max(0, y1 - pad)
        p_x2, p_y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        crop = frame[p_y1:p_y2, p_x1:p_x2]
        
        if crop.size == 0: return

        # Quality Metrics
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()
        
        # Basic quality check
        if blur_score < BLUR_THRESHOLD or brightness < DARKNESS_THRESHOLD:
            return

        is_good, q_reason = self._is_good_quality(None, face['landmark'])
        if not is_good: return

        # Combined Quality Score (Size + Clarity)
        quality_score = (face_w * face_h) / 1000.0 + (blur_score / 10.0)
        
        info['frame_count'] += 1
        
        if quality_score > info['best_quality']:
            info['best_quality'] = quality_score
            info['best_frame'] = frame.copy() # Store the full frame for better alignment later
            info['best_landmark'] = face['landmark'].copy()
            info['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Proactive trigger: If we've reached max frames
        if info['frame_count'] >= MAX_TRACKING_FRAMES:
            self._finalize_track(track_id)

    def _finalize_track(self, track_id):
        """Sends the aligned and enhanced best frame to the API."""
        info = self.trackers.get(track_id)
        if not info: return
        
        if info['best_frame'] is not None and not info['uploaded']:
            # 1. Align Face (CRITICAL for dlib accuracy)
            aligned_face = self._align_face(info['best_frame'], info['best_landmark'])
            
            # 2. Enhance for "Ultra Quality"
            enhanced = self._enhance_image(aligned_face)
            
            # 3. Send to API
            pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            self._send_to_api(pil_img, info['timestamp'], track_id)
            info['uploaded'] = True
            print(f"[SUCCESS] Tracker {track_id} finalized with Align+Enhance. Best Quality Score: {info['best_quality']:.2f}")

        del self.trackers[track_id]

    def _align_face(self, frame, landmark):
        """Aligns face based on eyes to ensure level features for dlib."""
        # Standard target points for a 256x256 crop (stable for dlib)
        desired_left_eye = (0.35, 0.35)
        desired_face_width = 256
        desired_face_height = 256

        # Landmark points: 0: left_eye, 1: right_eye
        left_eye_center = landmark[0]
        right_eye_center = landmark[1]

        # Calculate angle between eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate scale to reach desired eye distance
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = (1.0 - 2 * desired_left_eye[0]) * desired_face_width
        scale = desired_dist / dist

        # Center point between eyes (Float math for precision)
        eyes_center = (float(left_eye_center[0] + right_eye_center[0]) / 2.0,
                       float(left_eye_center[1] + right_eye_center[1]) / 2.0)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update translation
        t_x = desired_face_width * 0.5
        t_y = desired_face_height * desired_left_eye[1]
        M[0, 2] += (t_x - eyes_center[0])
        M[1, 2] += (t_y - eyes_center[1])

        # Apply Affine Transform
        aligned = cv2.warpAffine(frame, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)
        return aligned

    def _enhance_image(self, img):
        """Professional-grade image enhancement pipeline."""
        # 1. Denoising (Removes CCTV sensor grain)
        # Using a light fastNLMeans for performance while preserving edges
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        
        # 2. CLAHE (Local Contrast balancing)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)) # Lower limit to avoid haloing
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 3. High-Quality Sharpening (Unsharp Mask)
        # Subtracting a blurred version gives better results than ad-hoc kernels
        gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        enhanced = cv2.addWeighted(enhanced, 1.6, gaussian_blur, -0.6, 0)
        
        return enhanced

    def _is_good_quality(self, unused_crop, landmark: np.ndarray) -> Tuple[bool, str]:
        """Checks Pose and Head-Tilt Compensation."""
        if landmark is None or len(landmark) < 5:
            return False, "Insufficient landmarks"
            
        left_eye, right_eye, nose = landmark[0], landmark[1], landmark[2]
        left_mouth, right_mouth = landmark[3], landmark[4]
        
        # 1. Pose: Horizontal Symmetry (Frontal logic)
        l_dist = np.linalg.norm(left_eye - nose)
        r_dist = np.linalg.norm(right_eye - nose)
        
        if max(l_dist, r_dist) == 0:
            return False, "Invalid landmarks"
        
        symmetry_ratio = min(l_dist, r_dist) / max(l_dist, r_dist)
        if symmetry_ratio < POSE_THRESHOLD:
            return False, f"Side face (Symmetry: {symmetry_ratio:.2f} < {POSE_THRESHOLD})"
            
        # 2. Head-Tilt Compensation: Vertical Ratio
        eye_midpoint = (left_eye + right_eye) / 2
        mouth_midpoint = (left_mouth + right_mouth) / 2
        
        # Calculate vertical eye-to-nose vs nose-to-mouth ratio
        upper_dist = np.linalg.norm(eye_midpoint[1] - nose[1])
        lower_dist = np.linalg.norm(nose[1] - mouth_midpoint[1])
        
        # If the nose is almost touching the mouth (ratio < 0.2), they are looking too far down
        tilt_ratio = lower_dist / (upper_dist + 1e-6)
        if tilt_ratio < 0.2:
            return False, f"Looking down (Tilt ratio: {tilt_ratio:.2f} < 0.2)"
            
        return True, "Good"

    def _send_to_api(self, pil_img: Image.Image, timestamp: str, idx: int):
        """Saves face and uses fast OpenCV encoding for the API queue."""
        # Sanitize timestamp for Windows file system safety
        safe_timestamp = timestamp.replace(":", "-").replace(" ", "_")
        filename = f"Face_{safe_timestamp}_{idx}.jpg"
        path = self.faces_dir / filename
        
        try:
            # Convert PIL to fast OpenCV BGR format
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # Save locally
            cv2.imwrite(str(path), cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), API_JPEG_QUALITY])
            
            # FAST ENCODING: Use OpenCV to encode to memory
            success, buffer = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), API_JPEG_QUALITY])
            if not success:
                return

            # Queue data for background thread
            data = {
                'filename': filename,
                'content': buffer.tobytes(),
                'payload': {
                'camera_id': self.api_config['camera_id'],
                'device_id': self.api_config['device_id'],
                'device_name': self.api_config['device_name'],
                'org_id': self.api_config['org_id'],
                'captured_at': timestamp   
            }
            }
            
            try:
                self.api_queue.put_nowait(data)
                # Reduced logging to save I/O overhead
            except queue.Full:
                pass 
                
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
                
                if item is None:
                    break
                
                filename = item['filename']
                files = {'image': (filename, item['content'], 'image/jpeg')}
                
                try:
                    response = self.session.post(
                        self.api_config['api_url'], 
                        files=files, 
                        data=item['payload'], 
                        timeout=60
                    )
                    response_json = response.json() if response.status_code != 200 else {}
                    response.raise_for_status() 
                    print(f"[SUCCESS] API Backend confirmed receipt of {filename}!")
                    logging.info(f"Successfully sent {filename} to API.")
                except requests.exceptions.RequestException as e:
                    resp_text = e.response.text if hasattr(e, 'response') and e.response is not None else str(e)
                    error_msg = f"API Rejection for {filename}: {resp_text}"
                    print(f"[REJECTED] {error_msg}")
                    logging.error(error_msg)
                
                self.api_queue.task_done()
            except Exception as e:
                logging.error(f"Unexpected error in API worker thread: {e}")


def show_banner():
    """Displays a professional Hacker-style ASCII banner for the CLI."""
    banner = """
\033[92m╔══════════════════════════════════════════════════════════════════╗\033[0m
\033[92m║\033[0m                                                                  \033[92m║\033[0m
\033[92m║\033[0m    \033[92m______              _____      _ _           _\033[0m            \033[92m║\033[0m
\033[92m║\033[0m   \033[92m|  ____|            / ____|    | | |         | |\033[0m           \033[92m║\033[0m
\033[92m║\033[0m   \033[92m| |__ __ _  ___ ___| |     ___ | | | ___  ___| |_ ___  _ __\033[0m  \033[92m║\033[0m
\033[92m║\033[0m   \033[92m|  __/ _` |/ __/ _ \ |    / _ \| | |/ _ \/ __| __/ _ \| '__|\033[0m  \033[92m║\033[0m
\033[92m║\033[0m   \033[92m| | | (_| | (_|  __/ |___| (_) | | |  __/ (__| || (_) | |\033[0m    \033[92m║\033[0m
\033[92m║\033[0m   \033[92m|_|  \__,_|\___\___|\_____\___/|_|_|\___|\___|\__\___/|_|\033[0m    \033[92m║\033[0m
\033[92m║\033[0m                                                                  \033[92m║\033[0m
\033[92m╠══════════════════════════════════════════════════════════════════╣\033[0m
\033[92m║\033[0m  \033[1m[>] SYSTEM:\033[0m AI FACE COLLECTION & ALIGNMENT SYSTEM (CCTV)         \033[92m║\033[0m
\033[92m║\033[0m  \033[1m[>] VERSION:\033[0m 1.0.0-ULTRA (HACKER_EDITION)                        \033[92m║\033[0m
\033[92m║\033[0m  \033[1m[>] STATUS:\033[0m READY_TO_COLLECT                                    \033[92m║\033[0m
\033[92m╠══════════════════════════════════════════════════════════════════╣\033[0m
\033[92m║\033[0m  \033[1mCORE DEVELOPERED BY:\033[0m DATAMOO.AI              \033[92m║\033[0m
\033[92m╚══════════════════════════════════════════════════════════════════╝\033[0m
    """
    print(banner)

def show_footer():
    """Displays a professional Hacker-style shutdown banner."""
    footer = """
\033[92m╔══════════════════════════════════════════════════════════════════╗\033[0m
\033[92m║\033[0m  \033[91m[!] SYSTEM_HALTED\033[0m                                               \033[92m║\033[0m
\033[92m╠══════════════════════════════════════════════════════════════════╣\033[0m
\033[92m║\033[0m  \033[92mThank you for using Face Collector Ultra\033[0m                        \033[92m║\033[0m
\033[92m╚══════════════════════════════════════════════════════════════════╝\033[0m
    """
    print(footer)

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
    
    show_banner()
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
