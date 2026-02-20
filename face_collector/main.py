import cv2
import os
import time
import logging
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import requests
import io
import argparse
from facenet_pytorch import MTCNN

# ==========================================
# CONFIGURATION
# ==========================================
STREAM_URL = "http://192.168.68.103:8080/video"
OUTPUT_DIR = "output"
API_URL = "https://uatbase.amvipm.com/api/attendance"
PROCESS_INTERVAL = 0.2  # Seconds between processing frames
DETECTION_WIDTH = 640   # Resize width for detection speedup
MIN_FACE_SIZE = 60      # Minimum face width/height in pixels
BLUR_THRESHOLD = 80     # Laplancian variance threshold
POSE_THRESHOLD = 0.4    # Nose-to-eye ratio for frontal pose check
CONFIDENCE_THRESHOLD = 0.95 # MTCNN detection probability

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("face_collector.log")
    ]
)

class FaceCollector:
    def __init__(self, stream_url, base_output_dir):
        self.stream_url = stream_url
        self.faces_dir = os.path.join(base_output_dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")
        
        # MTCNN for detection & landmarks
        self.mtcnn = MTCNN(keep_all=True, device=self.device, select_largest=False)

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

    def _process_frame(self, frame):
        """Detects, aligns, and saves high-quality faces from a single frame."""
        # 1. Optimization: Resize for Detection
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
        except Exception as e:
            return # MTCNN internal error or empty

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

    def _is_good_quality(self, face_crop, landmark):
        """Checks Pose (Frontal logic)."""
        left_eye, right_eye, nose = landmark[0], landmark[1], landmark[2]
        
        # Pose: Ratio of nose distance to eyes
        l_dist = np.linalg.norm(left_eye - nose)
        r_dist = np.linalg.norm(right_eye - nose)
        
        if min(l_dist, r_dist) / max(l_dist, r_dist) < POSE_THRESHOLD:
            return False # Side profile
            
        return True

    def _align_face(self, face_crop, landmark):
        """Rotates face to make eyes horizontal."""
        left_eye, right_eye = landmark[0], landmark[1]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return face_crop.rotate(angle, expand=True)

    def _get_blur_score(self, pil_img):
        """Calculates Laplacian Variance."""
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _save_face(self, pil_img, timestamp, idx, score):
        filename = f"Face_{timestamp}_{idx}_S{int(score)}.jpg"
        path = os.path.join(self.faces_dir, filename)
        pil_img.save(path)
        logging.info(f"Saved: {filename}")

        # Send image to API
        try:
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            url = API_URL
            # Assuming the API expects the file in the 'image' field; adjust if it expects 'file', 'photo', etc.
            files = {'image': (filename, img_byte_arr, 'image/jpeg')}
            
            response = requests.post(url, files=files, timeout=10)
            if response.status_code in [200, 201]:
                logging.info(f"Successfully sent {filename} to API. Response: {response.text}")
            else:
                logging.warning(f"API returned status {response.status_code} for {filename}: {response.text}")
        except Exception as e:
            logging.error(f"Error sending {filename} to API: {e}")

def main():
    parser = argparse.ArgumentParser(description="Start the face collector.")
    parser.add_argument("--stream-url", type=str, default=STREAM_URL, help="URL of the video stream")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save output faces")
    args = parser.parse_args()
    
    collector = FaceCollector(args.stream_url, args.output_dir)
    collector.process_stream()

if __name__ == "__main__":
    main()
