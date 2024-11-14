import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk
import os
import torch
from ultralytics import YOLO
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.ABEnc import ABEnc
from threading import Thread
import queue
import logging

# ===========================
# Logging Configuration
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# ===========================
# CP-ABE Implementation
# ===========================
class CPabe_BSW07(ABEnc):
    def __init__(self, groupObj):
        ABEnc.__init__(self)
        self.util = SecretUtil(groupObj, verbose=False)
        self.group = groupObj

    def setup(self):
        g = self.group.random(G1)
        gp = self.group.random(G2)
        alpha = self.group.random(ZR)
        beta = self.group.random(ZR)
        h = g ** beta
        f = g ** (-beta)
        e_gg_alpha = pair(g, gp ** alpha)
        pk = {'g': g, 'g2': gp, 'h': h, 'f': f, 'e_gg_alpha': e_gg_alpha}
        mk = {'beta': beta, 'g2_alpha': gp ** alpha}
        return pk, mk

    def keygen(self, pk, mk, S):
        r = self.group.random()
        g_r = pk['g2'] ** r
        D = (mk['g2_alpha'] * g_r) ** mk['beta']
        D_j, D_j_pr = {}, {}
        for j in S:
            r_j = self.group.random()
            D_j[j] = g_r * (self.group.hash(j, G2) ** r_j)
            D_j_pr[j] = pk['g'] ** r_j
        return {'D': D, 'Dj': D_j, 'Djp': D_j_pr, 'S': S}

    def encrypt(self, pk, M, policy_str):
        policy = self.util.createPolicy(policy_str)
        s = self.group.random(ZR)
        C = pk['h'] ** s
        C_y, C_y_pr = {}, {}
        shares = self.util.calculateSharesDict(s, policy)

        for i in shares.keys():
            j = self.util.strip_index(i)
            C_y[i] = pk['g'] ** shares[i]
            C_y_pr[i] = self.group.hash(j, G2) ** shares[i]

        return {
            'C_tilde': (pk['e_gg_alpha'] ** s) * M,
            'C': C,
            'Cy': C_y,
            'Cyp': C_y_pr,
            'policy': policy_str,
        }

    def decrypt(self, pk, sk, ct):
        policy = self.util.createPolicy(ct['policy'])
        pruned_list = self.util.prune(policy, sk['S'])
        if pruned_list == False:
            return False
        z = self.util.getCoefficients(policy)
        A = 1
        for i in pruned_list:
            j = i.getAttributeAndIndex()
            k = i.getAttribute()
            A *= (pair(ct['Cy'][j], sk['Dj'][k]) / pair(sk['Djp'][k], ct['Cyp'][j])) ** z[j]
        return ct['C_tilde'] / (pair(ct['C'], sk['D']) / A)

# Initialize pairing group and CP-ABE
group = PairingGroup('SS512')
cpabe = CPabe_BSW07(group)
pk, mk = cpabe.setup()
logging.info("CP-ABE setup completed.")

# ===========================
# YOLOv8 Model Initialization
# ===========================
def initialize_yolo():
    try:
        # Load YOLOv8 models
        yolo_face = YOLO('/home/aditya/MTP3/yolov8n-face-lindevs.pt')  # Path to your face detection model
        yolo_plate = YOLO('/home/aditya/MTP3/best.pt')                # Path to your number plate detection model
        logging.info("YOLOv8 models loaded successfully.")
        
        # Verify class names
        logging.info("YOLO Face Model Classes: " + str(yolo_face.model.names))
        logging.info("YOLO Plate Model Classes: " + str(yolo_plate.model.names))
        
        return yolo_face, yolo_plate
    except Exception as e:
        logging.error(f"Failed to load YOLO models: {e}")
        messagebox.showerror("Error", f"Failed to load YOLO models:\n{str(e)}")
        return None, None

yolo_face, yolo_plate = initialize_yolo()
if yolo_face is None or yolo_plate is None:
    exit(1)  # Exit if models failed to load

# ===========================
# Tracker Initialization
# ===========================
def create_tracker_csrt():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    else:
        raise AttributeError("TrackerCSRT is not available in your OpenCV installation.")

def create_tracker_kcf():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerKCF_create()
    elif hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    else:
        raise AttributeError("TrackerKCF is not available in your OpenCV installation.")

# ===========================
# Detection Functions
# ===========================
def detect_faces(frame, conf_threshold=0.5):
    """
    Detect faces in the given frame using the YOLOv8 face model.
    Returns a list of bounding boxes: (x1, y1, x2, y2, confidence, class)
    """
    results = yolo_face(frame)
    detections = results[0].boxes  # Assuming single image input
    faces = []
    for box in detections:
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()
        cls_name = yolo_face.model.names[cls]
        if cls_name.lower() == 'face' and conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            faces.append((x1, y1, x2, y2, conf, cls))
            # Draw bounding box for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Face: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    logging.info(f"Detected {len(faces)} faces.")
    return faces

def detect_number_plates(frame, conf_threshold=0.5):
    """
    Detect number plates in the given frame using the YOLOv8 number plate model.
    Returns a list of bounding boxes: (x1, y1, x2, y2, confidence, class)
    """
    results = yolo_plate(frame)
    detections = results[0].boxes  # Assuming single image input
    plates = []
    for box in detections:
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()
        cls_name = yolo_plate.model.names[cls]
        if cls_name.lower() == 'number_plate' and conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plates.append((x1, y1, x2, y2, conf, cls))
            # Draw bounding box for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Plate: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    logging.info(f"Detected {len(plates)} number plates.")
    return plates

# ===========================
# Global Variables
# ===========================
policy_plate = 'ATTRIBUTEA'  # Policy for unblurring number plates
policy_face = 'ATTRIBUTEB'   # Policy for unblurring faces
hash_map = {}
ciphertext_map = {}
regions_map = {}             # Map: key -> (x, y, w, h)
trackers_map = {}            # Map: key -> tracker object
video_capture = None
blur_enabled_map = {}        # Map: key -> True/False
frame_queue = queue.Queue()  # Queue for thread-safe frame updates
retry_counters = {}          # Retry counters for trackers
max_retries = 5              # Maximum allowed consecutive tracker failures

# ===========================
# Video Processing Functions
# ===========================
def select_video():
    global video_capture, ciphertext_map, hash_map, regions_map, blur_enabled_map, trackers_map, retry_counters

    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if video_path:
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            messagebox.showerror("Error", "Cannot open the selected video.")
            logging.error("Failed to open video.")
            return

        # Read the first frame for detection
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read the video.")
            logging.error("Failed to read the first frame of the video.")
            return

        # Perform face and number plate detection
        plates = detect_number_plates(frame)
        faces = detect_faces(frame)

        # Reset previous data
        hash_map.clear()
        ciphertext_map.clear()
        regions_map.clear()
        blur_enabled_map.clear()
        trackers_map.clear()
        retry_counters.clear()

        # Encrypt and store number plate information
        if len(plates) > 0:
            for idx, (x1, y1, x2, y2, conf, cls) in enumerate(plates):
                if len(plates) == 1:
                    key = 'plate'
                else:
                    key = f'plate_{idx}'

                # Validate bounding box coordinates
                if x2 <= x1 or y2 <= y1:
                    logging.warning(f"Invalid bounding box for {key}: ({x1}, {y1}, {x2}, {y2})")
                    continue

                # Ensure bounding boxes are within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                w = x2 - x1
                h = y2 - y1

                if w < 30 or h < 30:  # Increased minimum size
                    logging.warning(f"Bounding box too small for {key}: width={w}, height={h}")
                    continue

                plate_region = frame[y1:y2, x1:x2]
                hash_value = group.hash(plate_region.tobytes(), ZR)
                M = pair(pk['g'], pk['g2']) ** hash_value
                hash_map[key] = M  # Store hash for verification
                ciphertext_map[key] = cpabe.encrypt(pk, M, policy_plate)
                regions_map[key] = (x1, y1, w, h)  # Store coordinates
                blur_enabled_map[key] = True  # Initialize blur flag
                retry_counters[key] = 0      # Initialize retry counter

                # Initialize tracker for this region
                try:
                    tracker = create_tracker_csrt()
                    tracker.init(frame, (x1, y1, w, h))
                    trackers_map[key] = tracker
                    logging.info(f"Initialized tracker for {key} with bbox: ({x1}, {y1}, {w}, {h})")
                except Exception as e:
                    messagebox.showerror("Error", f"Tracker initialization failed for {key}:\n{str(e)}")
                    logging.error(f"Tracker initialization failed for {key}: {e}")
                    continue

        # Encrypt and store face information
        if len(faces) > 0:
            for idx, (x1, y1, x2, y2, conf, cls) in enumerate(faces):
                if len(faces) == 1:
                    key = 'face'
                else:
                    key = f'face_{idx}'

                # Validate bounding box coordinates
                if x2 <= x1 or y2 <= y1:
                    logging.warning(f"Invalid bounding box for {key}: ({x1}, {y1}, {x2}, {y2})")
                    continue

                # Ensure bounding boxes are within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                w = x2 - x1
                h = y2 - y1

                if w < 30 or h < 30:  # Increased minimum size
                    logging.warning(f"Bounding box too small for {key}: width={w}, height={h}")
                    continue

                face_region = frame[y1:y2, x1:x2]
                hash_value = group.hash(face_region.tobytes(), ZR)
                M = pair(pk['g'], pk['g2']) ** hash_value
                hash_map[key] = M  # Store hash for verification
                ciphertext_map[key] = cpabe.encrypt(pk, M, policy_face)
                regions_map[key] = (x1, y1, w, h)  # Store coordinates
                blur_enabled_map[key] = True  # Initialize blur flag
                retry_counters[key] = 0      # Initialize retry counter

                # Initialize tracker for this region
                try:
                    tracker = create_tracker_csrt()
                    tracker.init(frame, (x1, y1, w, h))
                    trackers_map[key] = tracker
                    logging.info(f"Initialized tracker for {key} with bbox: ({x1}, {y1}, {w}, {h})")
                except Exception as e:
                    messagebox.showerror("Error", f"Tracker initialization failed for {key}:\n{str(e)}")
                    logging.error(f"Tracker initialization failed for {key}: {e}")
                    continue

        # Reset the video to start
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Start the video playback in a separate thread
        Thread(target=play_video, daemon=True).start()

def play_video():
    global video_capture, blur_enabled_map, trackers_map, regions_map, retry_counters

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_height, frame_width = frame.shape[:2]
        original_frame = frame.copy()      # For tracking
        processed_frame = frame.copy()     # For blurring and display

        # Update trackers
        keys_to_remove = []
        for key, tracker in list(trackers_map.items()):
            success, bbox = tracker.update(original_frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]

                # Validate coordinates
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                w = max(30, min(w, frame_width - x))  # Ensure minimum size
                h = max(30, min(h, frame_height - y))

                # Log bbox changes
                prev_bbox = regions_map.get(key, None)
                if prev_bbox:
                    logging.info(f"Previous bbox for '{key}': {prev_bbox}")
                logging.info(f"Current bbox for '{key}': ({x}, {y}, {w}, {h})")

                regions_map[key] = (x, y, w, h)
                retry_counters[key] = 0  # Reset retry counter
                logging.info(f"Tracker '{key}' successfully updated to bbox: ({x}, {y}, {w}, {h})")
            else:
                retry_counters[key] += 1
                logging.warning(f"Tracker '{key}' failed to update. Retry count: {retry_counters[key]}")
                if retry_counters[key] > max_retries:
                    keys_to_remove.append(key)
                    logging.warning(f"Tracker '{key}' exceeded max retries. Marking for removal.")

        # Remove failed trackers
        for key in keys_to_remove:
            del trackers_map[key]
            del regions_map[key]
            del hash_map[key]
            del ciphertext_map[key]
            del blur_enabled_map[key]
            del retry_counters[key]
            messagebox.showwarning("Tracker Lost", f"Tracker for '{key}' lost. Region has been removed.")
            logging.info(f"Tracker for '{key}' removed due to failure.")

        # Apply blurring
        for key, (x, y, w, h) in regions_map.items():
            if blur_enabled_map.get(key, False):
                # Ensure within frame boundaries
                if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
                    messagebox.showwarning("Invalid Coordinates", f"Coordinates for '{key}' are out of frame boundaries.")
                    logging.error(f"Invalid coordinates for '{key}': ({x}, {y}, {w}, {h})")
                    continue

                region = processed_frame[y:y+h, x:x+w]
                if region.size == 0:
                    messagebox.showwarning("Empty Region", f"Region '{key}' is empty. Skipping blurring.")
                    logging.warning(f"Empty region for '{key}'. Skipping blurring.")
                    continue

                try:
                    # Apply Gaussian Blur or Pixelation
                    blurred_region = cv2.GaussianBlur(region, (31, 31), 15)  # Reduced kernel size
                    # Alternatively, use pixelation
                    # blurred_region = pixelate_region(region, blocks=15)
                    processed_frame[y:y+h, x:x+w] = blurred_region
                    logging.info(f"Applied blur to '{key}' region.")
                    
                    # Draw bounding box for debugging
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(processed_frame, key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except cv2.error as e:
                    messagebox.showerror("Blurring Error", f"Failed to blur region '{key}': {str(e)}")
                    logging.error(f"Blurring failed for '{key}': {e}")

        # Enqueue the frame with bounding boxes for GUI
        frame_queue.put(processed_frame)

        # Increment frame count and perform detection at intervals
        frame_count = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        detection_interval = 30  # Detect every 30 frames (~1 second at 30 FPS)
        if frame_count % detection_interval == 0:
            detect_and_initialize_trackers(original_frame)

        # Control frame rate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_and_initialize_trackers(frame):
    global hash_map, ciphertext_map, regions_map, blur_enabled_map, trackers_map, retry_counters

    # Perform detections
    plates = detect_number_plates(frame)
    faces = detect_faces(frame)

    # Handle new number plates
    if len(plates) > 0:
        for idx, (x1, y1, x2, y2, conf, cls) in enumerate(plates):
            if len(plates) == 1:
                key = 'plate'
            else:
                key = f'plate_{idx}'

            # Validate bounding box coordinates
            if x2 <= x1 or y2 <= y1:
                logging.warning(f"Invalid bounding box for {key}: ({x1}, {y1}, {x2}, {y2})")
                continue

            # Ensure bounding boxes are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            w = x2 - x1
            h = y2 - y1

            if w < 30 or h < 30:  # Increased minimum size
                logging.warning(f"Bounding box too small for {key}: width={w}, height={h}")
                continue

            # Avoid initializing multiple trackers for the same key
            if key in trackers_map:
                continue

            plate_region = frame[y1:y2, x1:x2]
            hash_value = group.hash(plate_region.tobytes(), ZR)
            M = pair(pk['g'], pk['g2']) ** hash_value
            hash_map[key] = M  # Store hash for verification
            ciphertext_map[key] = cpabe.encrypt(pk, M, policy_plate)
            regions_map[key] = (x1, y1, w, h)  # Store coordinates
            blur_enabled_map[key] = True  # Initialize blur flag
            retry_counters[key] = 0      # Initialize retry counter

            # Initialize tracker for this region
            try:
                tracker = create_tracker_csrt()
                tracker.init(frame, (x1, y1, w, h))
                trackers_map[key] = tracker
                logging.info(f"Re-initialized tracker for {key} with bbox: ({x1}, {y1}, {w}, {h})")
            except Exception as e:
                messagebox.showerror("Error", f"Tracker initialization failed for {key}:\n{str(e)}")
                logging.error(f"Tracker initialization failed for {key}: {e}")
                continue

    # Handle new faces
    if len(faces) > 0:
        for idx, (x1, y1, x2, y2, conf, cls) in enumerate(faces):
            if len(faces) == 1:
                key = 'face'
            else:
                key = f'face_{idx}'

            # Validate bounding box coordinates
            if x2 <= x1 or y2 <= y1:
                logging.warning(f"Invalid bounding box for {key}: ({x1}, {y1}, {x2}, {y2})")
                continue

            # Ensure bounding boxes are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            w = x2 - x1
            h = y2 - y1

            if w < 30 or h < 30:  # Increased minimum size
                logging.warning(f"Bounding box too small for {key}: width={w}, height={h}")
                continue

            # Avoid initializing multiple trackers for the same key
            if key in trackers_map:
                continue

            face_region = frame[y1:y2, x1:x2]
            hash_value = group.hash(face_region.tobytes(), ZR)
            M = pair(pk['g'], pk['g2']) ** hash_value
            hash_map[key] = M  # Store hash for verification
            ciphertext_map[key] = cpabe.encrypt(pk, M, policy_face)
            regions_map[key] = (x1, y1, w, h)  # Store coordinates
            blur_enabled_map[key] = True  # Initialize blur flag
            retry_counters[key] = 0      # Initialize retry counter

            # Initialize tracker for this region
            try:
                tracker = create_tracker_csrt()
                tracker.init(frame, (x1, y1, w, h))
                trackers_map[key] = tracker
                logging.info(f"Re-initialized tracker for {key} with bbox: ({x1}, {y1}, {w}, {h})")
            except Exception as e:
                messagebox.showerror("Error", f"Tracker initialization failed for {key}:\n{str(e)}")
                logging.error(f"Tracker initialization failed for {key}: {e}")
                continue

# ===========================
# GUI Update Function
# ===========================
def update_gui():
    if not frame_queue.empty():
        frame = frame_queue.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        label_video.config(image=imgtk)
        label_video.image = imgtk
    root.after(10, update_gui)  # Adjust the delay as needed

# ===========================
# Unblurring Functions
# ===========================
def unblur_number_plate():
    # Determine the relevant key based on the number of plates detected
    plate_keys = [key for key in hash_map.keys() if key.startswith('plate')]

    if not plate_keys:
        messagebox.showerror("Error", "No video selected or number plate not detected.")
        logging.error("Unblur attempt failed: No number plates detected.")
        return

    if len(plate_keys) == 1:
        # Only one number plate detected
        selected_key = 'plate'
    else:
        # Multiple number plates detected, prompt user to select which one to unblur
        plate_selection = simpledialog.askinteger(
            "Select Number Plate",
            f"Enter the number plate number to unblur (1 to {len(plate_keys)}):",
            minvalue=1,
            maxvalue=len(plate_keys)
        )

        if not plate_selection:
            return  # User cancelled the dialog

        selected_key = f'plate_{plate_selection - 1}'

    unblur_region(selected_key, policy_plate)

def unblur_face():
    # Determine the relevant key based on the number of faces detected
    face_keys = [key for key in hash_map.keys() if key.startswith('face')]

    if not face_keys:
        messagebox.showerror("Error", "No video selected or face not detected.")
        logging.error("Unblur attempt failed: No faces detected.")
        return

    if len(face_keys) == 1:
        # Only one face detected
        selected_key = 'face'
    else:
        # Multiple faces detected, prompt user to select which one to unblur
        face_selection = simpledialog.askinteger(
            "Select Face",
            f"Enter the face number to unblur (1 to {len(face_keys)}):",
            minvalue=1,
            maxvalue=len(face_keys)
        )

        if not face_selection:
            return  # User cancelled the dialog

        selected_key = f'face_{face_selection - 1}'

    unblur_region(selected_key, policy_face)

def unblur_region(region_key, policy):
    global pk, mk, hash_map, ciphertext_map, blur_enabled_map

    if not hash_map.get(region_key) or region_key not in ciphertext_map or not ciphertext_map[region_key]:
        messagebox.showerror("Error", f"No video selected or {region_key.replace('_', ' ').capitalize()} not detected.")
        logging.error(f"Unblur attempt failed: {region_key} not detected or encrypted.")
        return

    attributes = select_attributes()
    if not attributes:
        return

    attribute_list = [attr.strip().upper() for attr in attributes]

    # Generate user's secret key
    try:
        sk = cpabe.keygen(pk, mk, attribute_list)
        logging.info(f"Secret key generated for attributes: {attribute_list}")
    except Exception as e:
        messagebox.showerror("Error", f"Key generation failed:\n{str(e)}")
        logging.error(f"Key generation failed for {region_key}: {e}")
        return

    # Attempt to decrypt the ciphertext
    try:
        ciphertext = ciphertext_map.get(region_key, None)
        if not ciphertext:
            messagebox.showerror("Error", "Ciphertext not found.")
            logging.error(f"Ciphertext not found for {region_key}.")
            return

        rec_msg = cpabe.decrypt(pk, sk, ciphertext)
        logging.info(f"Decryption attempt for {region_key} successful.")
    except Exception as e:
        messagebox.showerror("Error", f"Decryption failed:\n{str(e)}")
        logging.error(f"Decryption failed for {region_key}: {e}")
        return

    # Verify the decrypted message against the stored hash
    expected_msg = hash_map.get(region_key, None)
    if rec_msg and expected_msg and group.serialize(rec_msg) == group.serialize(expected_msg):
        blur_enabled_map[region_key] = False  # Disable blurring for this specific region
        messagebox.showinfo("Success", f"{region_key.replace('_', ' ').capitalize()} successfully unblurred.")
        logging.info(f"{region_key} successfully unblurred.")
    else:
        messagebox.showerror("Error", "Incorrect attributes or decryption failed.")
        logging.warning(f"Unblurring failed for {region_key}: Incorrect attributes or decryption mismatch.")

def select_attributes():
    # Provide a dialog with checkboxes for predefined attributes
    attributes = ["ATTRIBUTEA", "ATTRIBUTEB", "ATTRIBUTEC"]  # Example attributes
    selected = []

    def on_select():
        for var, attr in zip(vars, attributes):
            if var.get():
                selected.append(attr)
        dialog.destroy()

    dialog = tk.Toplevel(root)
    dialog.title("Select Attributes")
    tk.Label(dialog, text="Select your attributes:").pack(pady=10)

    vars = [tk.IntVar() for _ in attributes]
    for var, attr in zip(vars, attributes):
        tk.Checkbutton(dialog, text=attr, variable=var).pack(anchor='w')

    tk.Button(dialog, text="Submit", command=on_select).pack(pady=10)
    dialog.grab_set()
    root.wait_window(dialog)
    return selected

# ===========================
# GUI Setup
# ===========================
def main():
    global root, label_video
    root = tk.Tk()
    root.title("Video Number Plate and Face Blurring Tool")
    root.geometry("1000x700")

    # Select Video Button
    btn_select = tk.Button(root, text="Select Video", command=select_video)
    btn_select.pack(pady=10)

    # Video Display Label
    label_video = tk.Label(root)
    label_video.pack(side=tk.TOP, pady=10)

    # Unblur Buttons
    btn_unblur_plate = tk.Button(root, text="Unblur Number Plate", command=unblur_number_plate)
    btn_unblur_plate.pack(pady=10)

    btn_unblur_face = tk.Button(root, text="Unblur Face", command=unblur_face)
    btn_unblur_face.pack(pady=10)

    # Status Label
    status_label = tk.Label(root, text="Status: Ready", anchor='w')
    status_label.pack(fill='x', padx=10, pady=5)

    # Start the GUI update loop
    root.after(0, update_gui)

    root.mainloop()

    # Release resources after the window is closed
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
