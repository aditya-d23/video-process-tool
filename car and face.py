import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk
import os
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.ABEnc import ABEnc
from threading import Thread

# Initialize pairing group for CP-ABE
group = PairingGroup('SS512')

# Placeholder for global variables
cpabe = None
pk = None
mk = None
policy_plate = 'attributeA'  # Policy for unblurring number plates
policy_face = 'attributeB'   # Policy for unblurring faces
hash_map = {}
ciphertext_map = {}
regions_map = {}             # Map: key -> (x, y, w, h)
trackers_map = {}            # Map: key -> tracker object
video_capture = None
blur_enabled_map = {}        # Map: key -> True/False

# Initialize CP-ABE scheme
def initialize_cpabe():
    global cpabe, pk, mk
    cpabe = CPabe_BSW07(group)
    (pk, mk) = cpabe.setup()

class CPabe_BSW07(ABEnc):
    def __init__(self, groupObj):
        ABEnc.__init__(self)
        global util, group
        util = SecretUtil(groupObj, verbose=False)
        group = groupObj

    def setup(self):
        g, gp = group.random(G1), group.random(G2)
        alpha, beta = group.random(ZR), group.random(ZR)
        h = g ** beta
        f = g ** (~beta)
        e_gg_alpha = pair(g, gp ** alpha)
        pk = {'g': g, 'g2': gp, 'h': h, 'f': f, 'e_gg_alpha': e_gg_alpha}
        mk = {'beta': beta, 'g2_alpha': gp ** alpha}
        return (pk, mk)

    def keygen(self, pk, mk, S):
        r = group.random()
        g_r = pk['g2'] ** r
        D = (mk['g2_alpha'] * g_r) ** (1 / mk['beta'])
        D_j, D_j_pr = {}, {}
        for j in S:
            r_j = group.random()
            D_j[j] = g_r * (group.hash(j, G2) ** r_j)
            D_j_pr[j] = pk['g'] ** r_j
        return {'D': D, 'Dj': D_j, 'Djp': D_j_pr, 'S': S}

    def encrypt(self, pk, M, policy_str):
        policy = util.createPolicy(policy_str)
        s = group.random(ZR)
        C = pk['h'] ** s
        C_y, C_y_pr = {}, {}
        shares = util.calculateSharesDict(s, policy)

        for i in shares.keys():
            j = util.strip_index(i)
            C_y[i] = pk['g'] ** shares[i]
            C_y_pr[i] = group.hash(j, G2) ** shares[i]

        return {
            'C_tilde': (pk['e_gg_alpha'] ** s) * M,
            'C': C,
            'Cy': C_y,
            'Cyp': C_y_pr,
            'policy': policy_str,
        }

    def decrypt(self, pk, sk, ct):
        policy = util.createPolicy(ct['policy'])
        pruned_list = util.prune(policy, sk['S'])
        if pruned_list == False:
            return False
        z = util.getCoefficients(policy)
        A = 1
        for i in pruned_list:
            j = i.getAttributeAndIndex()
            k = i.getAttribute()
            A *= (pair(ct['Cy'][j], sk['Dj'][k]) / pair(sk['Djp'][k], ct['Cyp'][j])) ** z[j]
        return ct['C_tilde'] / (pair(ct['C'], sk['D']) / A)

# Function to create CSRT tracker
def create_tracker_csrt():
    # For OpenCV 4.x with contrib modules
    return cv2.legacy.TrackerCSRT_create()

# Tkinter GUI Functions
def select_video():
    global video_capture, ciphertext_map, hash_map, regions_map, blur_enabled_map, trackers_map

    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if video_path:
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            messagebox.showerror("Error", "Cannot open the selected video.")
            return

        # Load the pre-trained Haar Cascade classifiers for number plate and face detection
        cascade_plate_path = 'haarcascade_russian_plate_number.xml'
        cascade_face_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_plate_path) or not os.path.exists(cascade_face_path):
            messagebox.showerror("Error", f"Haar Cascade files not found.\nEnsure '{cascade_plate_path}' and '{cascade_face_path}' are in the script directory.")
            return

        number_plate_cascade = cv2.CascadeClassifier(cascade_plate_path)
        face_cascade = cv2.CascadeClassifier(cascade_face_path)

        # Initialize the video frame for hashing and encryption
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read the video.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = number_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

        # Reset previous data
        hash_map.clear()
        ciphertext_map.clear()
        regions_map.clear()
        blur_enabled_map.clear()
        trackers_map.clear()

        # Encrypt and store number plate information
        if len(plates) > 0:
            for idx, (x, y, w, h) in enumerate(plates):
                if len(plates) == 1:
                    key = 'plate'
                else:
                    key = f'plate_{idx}'
                plate_region = frame[y:y+h, x:x+w]
                hash_value = group.hash(plate_region.tobytes(), ZR)
                M = pair(pk['g'], pk['g2']) ** hash_value
                hash_map[key] = M  # Store hash for verification
                ciphertext_map[key] = cpabe.encrypt(pk, M, policy_plate)
                regions_map[key] = (x, y, w, h)  # Store coordinates
                blur_enabled_map[key] = True  # Initialize blur flag

                # Initialize tracker for this region
                try:
                    tracker = create_tracker_csrt()
                    tracker.init(frame, (x, y, w, h))
                    trackers_map[key] = tracker
                except AttributeError as e:
                    messagebox.showerror("Error", str(e))
                    return

        # Encrypt and store face information
        if len(faces) > 0:
            for idx, (x, y, w, h) in enumerate(faces):
                if len(faces) == 1:
                    key = 'face'
                else:
                    key = f'face_{idx}'
                face_region = frame[y:y+h, x:x+w]
                hash_value = group.hash(face_region.tobytes(), ZR)
                M = pair(pk['g'], pk['g2']) ** hash_value
                hash_map[key] = M  # Store hash for verification
                ciphertext_map[key] = cpabe.encrypt(pk, M, policy_face)
                regions_map[key] = (x, y, w, h)  # Store coordinates
                blur_enabled_map[key] = True  # Initialize blur flag

                # Initialize tracker for this region
                try:
                    tracker = create_tracker_csrt()
                    tracker.init(frame, (x, y, w, h))
                    trackers_map[key] = tracker
                except AttributeError as e:
                    messagebox.showerror("Error", str(e))
                    return

        # Reset the video to start
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Start the video playback in a separate thread
        Thread(target=play_video, daemon=True).start()

def play_video():
    global video_capture, blur_enabled_map, trackers_map, regions_map

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_height, frame_width = frame.shape[:2]

        # Update trackers and get new positions
        keys_to_remove = []
        for key, tracker in trackers_map.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]

                # Validate coordinates
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                w = max(1, min(w, frame_width - x))
                h = max(1, min(h, frame_height - y))

                regions_map[key] = (x, y, w, h)
            else:
                # Tracker failed; mark key for removal
                keys_to_remove.append(key)

        # Remove failed trackers and their associated data
        for key in keys_to_remove:
            del trackers_map[key]
            del regions_map[key]
            del hash_map[key]
            del ciphertext_map[key]
            del blur_enabled_map[key]
            messagebox.showwarning("Tracker Lost", f"Tracker for '{key}' lost. Region has been removed.")

        # Apply blurring based on blur_enabled_map
        for key, (x, y, w, h) in regions_map.items():
            if blur_enabled_map.get(key, False):
                # Ensure coordinates are within frame boundaries
                if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
                    messagebox.showwarning("Invalid Coordinates", f"Coordinates for '{key}' are out of frame boundaries.")
                    continue  # Skip blurring for this region

                # Extract the region of interest (ROI)
                region = frame[y:y+h, x:x+w]

                # Check if the region is not empty
                if region.size == 0:
                    messagebox.showwarning("Empty Region", f"Region '{key}' is empty. Skipping blurring.")
                    continue  # Skip blurring for this region

                # Apply Gaussian Blur
                try:
                    blurred_region = cv2.GaussianBlur(region, (99, 99), 30)
                    frame[y:y+h, x:x+w] = blurred_region
                except cv2.error as e:
                    messagebox.showerror("Blurring Error", f"Failed to blur region '{key}': {str(e)}")
                    continue  # Skip blurring for this region

        # Convert frame for Tkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)

        # Update the Tkinter label with the new frame
        label_video.config(image=imgtk)
        label_video.image = imgtk

        # Delay for a short period to control frame rate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def unblur_number_plate():
    # Determine the relevant key based on the number of plates detected
    plate_keys = [key for key in hash_map.keys() if key.startswith('plate')]

    if not plate_keys:
        messagebox.showerror("Error", "No video selected or number plate not detected.")
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
        return

    attributes = simpledialog.askstring("Input", "Enter your attributes separated by commas (e.g., ATTRIBUTEA, ATTRIBUTEB):")
    if not attributes:
        return

    attribute_list = [attr.strip().upper() for attr in attributes.split(",")]

    # Generate user's secret key
    try:
        sk = cpabe.keygen(pk, mk, attribute_list)
    except Exception as e:
        messagebox.showerror("Error", f"Key generation failed:\n{str(e)}")
        return

    # Attempt to decrypt the ciphertext
    try:
        ciphertext = ciphertext_map.get(region_key, None)
        if not ciphertext:
            messagebox.showerror("Error", "Ciphertext not found.")
            return

        rec_msg = cpabe.decrypt(pk, sk, ciphertext)
    except Exception as e:
        messagebox.showerror("Error", f"Decryption failed:\n{str(e)}")
        return

    # Verify the decrypted message against the stored hash
    expected_msg = hash_map.get(region_key, None)
    if rec_msg and expected_msg and group.serialize(rec_msg) == group.serialize(expected_msg):
        blur_enabled_map[region_key] = False  # Disable blurring for this specific region
        messagebox.showinfo("Success", f"{region_key.replace('_', ' ').capitalize()} successfully unblurred.")
    else:
        messagebox.showerror("Error", "Incorrect attributes or decryption failed.")

def main():
    initialize_cpabe()

    # Tkinter setup
    root = tk.Tk()
    root.title("Video Number Plate and Face Blurring Tool")
    root.geometry("1000x700")

    btn_select = tk.Button(root, text="Select Video", command=select_video)
    btn_select.pack(pady=10)

    global label_video
    label_video = tk.Label(root)
    label_video.pack(side=tk.TOP, pady=10)

    btn_unblur_plate = tk.Button(root, text="Unblur Number Plate", command=unblur_number_plate)
    btn_unblur_plate.pack(pady=10)

    btn_unblur_face = tk.Button(root, text="Unblur Face", command=unblur_face)
    btn_unblur_face.pack(pady=10)

    root.mainloop()

    # Release resources after the window is closed
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
