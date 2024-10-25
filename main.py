import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk
import os
import time
from charm.toolbox.pairinggroup import PairingGroup, G1, G2, GT, ZR, pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.ABEnc import ABEnc
import threading
from tkinter import ttk  # For progress bar

# ---------------------------- CP-ABE Implementation ---------------------------- #

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
        return (pk, mk)

    def keygen(self, pk, mk, S):
        r = self.group.random()
        g_r = pk['g2'] ** r
        D = (mk['g2_alpha'] * g_r) ** (1 / mk['beta'])
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

        C_tilde = (pk['e_gg_alpha'] ** s) * M
        ciphertext = {
            'C_tilde': C_tilde,
            'C': C,
            'Cy': C_y,
            'Cyp': C_y_pr,
            'policy': policy_str,
        }
        return ciphertext

    def decrypt(self, pk, sk, ct):
        policy = self.util.createPolicy(ct['policy'])
        pruned_list = self.util.prune(policy, sk['S'])
        if pruned_list is False:
            return False
        z = self.util.getCoefficients(policy)
        A = 1
        for i in pruned_list:
            j = i.getAttributeAndIndex()
            k = i.getAttribute()
            A *= (pair(ct['Cy'][j], sk['Dj'][k]) / pair(sk['Djp'][k], ct['Cyp'][j])) ** z[j]
        decrypted = ct['C_tilde'] / (pair(ct['C'], sk['D']) / A)
        return decrypted

# --------------------------- Video Blurring Tool ------------------------------ #

class VideoBlurringTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Blurring Tool with CP-ABE")
        self.root.geometry("1200x800")

        # Initialize CP-ABE
        try:
            self.group = PairingGroup('SS512')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize pairing group:\n{str(e)}")
            self.root.destroy()
            return

        try:
            self.cpabe = CPabe_BSW07(self.group)
            self.pk, self.mk = self.cpabe.setup()
        except Exception as e:
            messagebox.showerror("Error", f"Failed during CP-ABE setup:\n{str(e)}")
            self.root.destroy()
            return

        # Define policies with simplified attribute names and lowercase operators
        self.policies = {
            'face': [
                '((A and B) or C)', 
                'D', 
                'E'
            ],
            'number_plate': [
                'X and Y',
                'Z'
            ]
        }

        # Initialize maps to store hashes and ciphertexts
        self.hash_map = {
            'face': {},
            'number_plate': {}
        }
        self.ciphertext_map = {
            'face': [],
            'number_plate': []
        }

        # Lists to store regions
        self.face_regions = []
        self.number_plate_regions = []

        # Dictionary to store original regions for potential unblurring
        self.original_regions = {
            'face': {},
            'number_plate': {}
        }

        # Initialize detectors
        self.face_net = None
        self.plate_cascade = None
        self.load_detectors()

        # Setup GUI
        self.setup_gui()

        # Video Playback Variables
        self.video_running = False
        self.cap = None
        self.output_path = None

    def load_detectors(self):
        # Load face detector
        prototxt_path = '/home/aditya/MTP3/deploy.prototxt.txt'
        model_path = '/home/aditya/MTP3/res10_300x300_ssd_iter_140000.caffemodel'
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            messagebox.showerror("Error", f"Face model files not found.\nExpected '{prototxt_path}' and '{model_path}' in the script directory.")
            self.root.destroy()
            return
        try:
            self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load face detection model:\n{str(e)}")
            self.root.destroy()
            return

        # Load number plate detector
        plate_cascade_path = '/home/aditya/MTP3/haarcascade_russian_plate_number.xml'
        if not os.path.exists(plate_cascade_path):
            messagebox.showerror("Error", f"Number plate cascade file not found at '{plate_cascade_path}'. Please download it and place it in the script directory.")
            self.root.destroy()
            return
        try:
            self.plate_cascade = cv2.CascadeClassifier(plate_cascade_path)
            if self.plate_cascade.empty():
                raise ValueError("Loaded cascade is empty.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load number plate detection cascade:\n{str(e)}")
            self.root.destroy()
            return

    def setup_gui(self):
        # Video selection
        btn_select_video = tk.Button(self.root, text="Select Video", command=self.select_video, width=20, height=2)
        btn_select_video.pack(pady=20)

        # Video display
        self.label_video = tk.Label(self.root)
        self.label_video.pack(side=tk.TOP, pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

        # Processing controls
        btn_process_video = tk.Button(self.root, text="Play Video", command=self.process_selected_video, width=20, height=2)
        btn_process_video.pack(pady=20)

        # Unblurring controls
        frame_unblur = tk.Frame(self.root)
        frame_unblur.pack(side=tk.BOTTOM, fill=tk.X, pady=10, anchor='s')

        btn_unblur_face = tk.Button(frame_unblur, text="Unblur Face", command=self.unblur_face_gui, width=15, height=2)
        btn_unblur_face.pack(side=tk.LEFT, padx=10, pady=5)

        btn_unblur_plate = tk.Button(frame_unblur, text="Unblur Number Plate", command=self.unblur_plate_gui, width=20, height=2)
        btn_unblur_plate.pack(side=tk.LEFT, padx=10, pady=5)

        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor='w')
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def select_video(self):
        # Allow selecting any video file type
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All Video Files", "*.*"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
                ("FLV files", "*.flv"),
                ("WMV files", "*.wmv"),
                ("MPEG files", "*.mpeg;*.mpg"),
                ("M4V files", "*.m4v")
            ]
        )
        if file_path:
            self.selected_video_path = file_path
            messagebox.showinfo("Selected Video", f"Selected video:\n{file_path}")
            print(f"Selected video: {file_path}")
            self.status_label.config(text=f"Selected video: {os.path.basename(file_path)}")
        else:
            messagebox.showwarning("No Selection", "No video file selected.")
            print("No video file selected.")
            self.status_label.config(text="No video selected.")

    def get_fourcc(self, output_path):
        """
        Determines the appropriate fourcc code based on the output file extension.
        """
        extension = os.path.splitext(output_path)[1].lower()
        codec_map = {
            '.mp4': 'mp4v',
            '.m4v': 'mp4v',
            '.avi': 'XVID',
            '.mov': 'avc1',
            '.mkv': 'X264',
            '.flv': 'FLV1',
            '.wmv': 'WMV1',
            '.mpeg': 'MPEG',
            '.mpg': 'MPEG'
        }
        fourcc_str = codec_map.get(extension, 'XVID')  # Default to 'XVID' if extension not found
        print(f"Determined FourCC: {fourcc_str} for extension: {extension}")
        return cv2.VideoWriter_fourcc(*fourcc_str)

    def process_selected_video(self):
        if not hasattr(self, 'selected_video_path'):
            messagebox.showerror("Error", "No video selected. Please select a video first.")
            print("Processing aborted: No video selected.")
            return

        # Ask user for output video path
        output_path = filedialog.asksaveasfilename(
            defaultextension="",  # Let the user specify the extension
            filetypes=[("All files", "*.*")]
        )
        if not output_path:
            messagebox.showwarning("No Output Path", "No output path specified. Processing aborted.")
            print("Processing aborted: No output path specified.")
            return

        self.output_path = output_path

        # Start processing and playback in a separate thread
        processing_thread = threading.Thread(target=self.play_video, args=(self.selected_video_path, self.output_path))
        processing_thread.start()

    def play_video(self, video_path, output_path):
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open video file: {video_path}")
                print(f"Failed to open video file: {video_path}")
                return

            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  

            # Determine fourcc based on output file extension
            fourcc = self.get_fourcc(output_path)
            print(f"Selected FourCC: {fourcc}")

            # Initialize video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                messagebox.showerror("Error", f"Cannot write to video file: {output_path}")
                print(f"Failed to initialize video writer: {output_path}")
                self.cap.release()
                return

            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            # Configure progress bar
            self.progress['maximum'] = frame_count
            self.progress['value'] = 0
            self.status_label.config(text="Playing video...")

            self.video_running = True
            while self.video_running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                out.write(processed_frame)
                self.display_frame(processed_frame)

                current_frame += 1
                if current_frame % 10 == 0 or current_frame == frame_count:
                    print(f'Processing frame {current_frame}/{frame_count}')
                    self.progress['value'] = current_frame
                    self.status_label.config(text=f"Processing frame {current_frame}/{frame_count}")
                    self.root.update_idletasks()

                # Wait for the appropriate time based on FPS
                delay = int(1000 / fps) if fps > 0 else 33  # Default to ~30 FPS
                if delay > 0:
                    self.root.after(delay)

            # Release resources
            self.cap.release()
            out.release()
            self.progress['value'] = frame_count
            self.status_label.config(text="Video processing complete.")
            messagebox.showinfo("Success", f"Processed video saved to:\n{output_path}")
            print("Video processing complete.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during video processing:\n{str(e)}")
            print(f"An error occurred during video processing: {str(e)}")

    def display_frame(self, frame):
        # Convert the frame to RGB and then to PIL Image
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_video.imgtk = imgtk  # Prevent garbage collection
        self.label_video.configure(image=imgtk)

    def process_frame(self, frame):
        resized_frame = self.resize_image(frame, width=800, height=600)
        original_frame = resized_frame.copy()
        blurred_frame = resized_frame.copy()

        h, w = resized_frame.shape[:2]

        # Detect faces
        faces = self.detect_faces(resized_frame)

        # Detect number plates
        number_plates = self.detect_number_plates(resized_frame)

        # Process faces
        for face in faces:
            (x, y, x1, y1) = face
            face_roi = blurred_frame[y:y1, x:x1]
            face_blur = cv2.GaussianBlur(face_roi, (99, 99), 30)
            blurred_frame[y:y1, x:x1] = face_blur
            self.face_regions.append((x, y, x1, y1))

            # Compute hash and encrypt
            original_face_region = original_frame[y:y1, x:x1]
            hash_value = self.group.hash(original_face_region.tobytes(), ZR)
            M = pair(self.pk['g'], self.pk['g2']) ** hash_value

            face_index = len(self.face_regions)
            self.hash_map['face'][face_index] = M
            policy_str = self.policies['face'][(face_index - 1) % len(self.policies['face'])]
            try:
                ct = self.cpabe.encrypt(self.pk, M, policy_str)
                self.ciphertext_map['face'].append(ct)
                self.original_regions['face'][face_index] = original_face_region.copy()
                print(f"Encrypted face {face_index} with policy: {policy_str}")
            except Exception as e:
                print(f"Encryption failed for face {face_index}: {str(e)}")

        # Process number plates
        for plate in number_plates:
            (x, y, x1, y1) = plate
            plate_roi = blurred_frame[y:y1, x:x1]
            plate_blur = cv2.GaussianBlur(plate_roi, (99, 99), 30)
            blurred_frame[y:y1, x:x1] = plate_blur
            self.number_plate_regions.append((x, y, x1, y1))

            # Compute hash and encrypt
            original_plate_region = original_frame[y:y1, x:x1]
            hash_value = self.group.hash(original_plate_region.tobytes(), ZR)
            M = pair(self.pk['g'], self.pk['g2']) ** hash_value

            plate_index = len(self.number_plate_regions)
            self.hash_map['number_plate'][plate_index] = M
            policy_str = self.policies['number_plate'][(plate_index - 1) % len(self.policies['number_plate'])]
            try:
                ct = self.cpabe.encrypt(self.pk, M, policy_str)
                self.ciphertext_map['number_plate'].append(ct)
                self.original_regions['number_plate'][plate_index] = original_plate_region.copy()
                print(f"Encrypted number plate {plate_index} with policy: {policy_str}")
            except Exception as e:
                print(f"Encryption failed for number plate {plate_index}: {str(e)}")

        return blurred_frame

    def detect_faces(self, frame, conf_threshold=0.5):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, x1, y1) = box.astype("int")
                x, y, x1, y1 = max(0, x), max(0, y), min(frame.shape[1], x1), min(frame.shape[0], y1)
                if x < x1 and y < y1:
                    faces.append((x, y, x1, y1))
        return faces

    def detect_number_plates(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        number_plates = []
        for (x, y, w, h) in plates:
            number_plates.append((x, y, x + w, y + h))
        return number_plates

    def resize_image(self, image, width=800, height=600):
        h, w = image.shape[:2]
        aspect_ratio = w / h
        if (width / height) > aspect_ratio:
            new_height = height
            new_width = int(aspect_ratio * height)
        else:
            new_width = width
            new_height = int(width / aspect_ratio)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image

    def unblur_face_gui(self):
        if not self.face_regions:
            messagebox.showinfo("Info", "No faces to unblur.")
            print("Unblur Face: No faces detected.")
            return

        face_number = simpledialog.askinteger("Input", "Enter face number to unblur:")
        if not face_number or face_number < 1 or face_number > len(self.face_regions):
            messagebox.showerror("Error", "Invalid face number.")
            print(f"Unblur Face: Invalid face number entered: {face_number}")
            return

        attributes = simpledialog.askstring("Input", f"Enter attributes for Face {face_number} separated by commas:")
        if not attributes:
            messagebox.showwarning("Warning", "No attributes entered. Unblurring aborted.")
            print("Unblur Face: No attributes entered.")
            return

        # Decrypt and verify
        threading.Thread(target=self.unblur_region, args=('face', face_number, attributes)).start()

    def unblur_plate_gui(self):
        if not self.number_plate_regions:
            messagebox.showinfo("Info", "No number plates to unblur.")
            print("Unblur Number Plate: No number plates detected.")
            return

        plate_number = simpledialog.askinteger("Input", "Enter number plate number to unblur:")
        if not plate_number or plate_number < 1 or plate_number > len(self.number_plate_regions):
            messagebox.showerror("Error", "Invalid number plate number.")
            print(f"Unblur Number Plate: Invalid number plate number entered: {plate_number}")
            return

        attributes = simpledialog.askstring("Input", f"Enter attributes for Number Plate {plate_number} separated by commas:")
        if not attributes:
            messagebox.showwarning("Warning", "No attributes entered. Unblurring aborted.")
            print("Unblur Number Plate: No attributes entered.")
            return

        # Decrypt and verify
        threading.Thread(target=self.unblur_region, args=('number_plate', plate_number, attributes)).start()

    def unblur_region(self, region_type, region_number, attributes):
        try:
            ct = self.ciphertext_map[region_type][region_number - 1]
        except IndexError:
            messagebox.showerror("Error", f"No ciphertext found for {region_type} {region_number}.")
            print(f"Unblur {region_type.title()}: Ciphertext not found for {region_type} {region_number}.")
            return

        attribute_list = [attr.strip().upper() for attr in attributes.split(",")]
        try:
            sk = self.cpabe.keygen(self.pk, self.mk, attribute_list)
            print(f"Generated secret key for attributes: {attribute_list}")
        except Exception as e:
            messagebox.showerror("Error", f"Key generation failed:\n{str(e)}")
            print(f"Unblur {region_type.title()}: Key generation failed with error: {str(e)}")
            return

        try:
            rec_msg = self.cpabe.decrypt(self.pk, sk, ct)
            print(f"Decryption result for {region_type} {region_number}: {rec_msg}")
        except Exception as e:
            messagebox.showerror("Error", f"Decryption failed:\n{str(e)}")
            print(f"Unblur {region_type.title()}: Decryption failed with error: {str(e)}")
            return

        expected_msg = self.hash_map[region_type].get(region_number, None)
        if rec_msg and self.group.serialize(rec_msg) == self.group.serialize(expected_msg):
            # Unblur the region by restoring the original region
            # Since we are processing frames in real-time, we'll need to keep track of which regions to unblur
            # For simplicity, we'll mark the region as unblurred and skip blurring it in future frames
            if region_number in self.original_regions[region_type]:
                # Remove the region from the list to prevent blurring in future frames
                self.mark_region_unblurred(region_type, region_number)
                messagebox.showinfo("Success", f"{region_type.replace('_', ' ').title()} {region_number} successfully unblurred.")
                print(f"Unblur {region_type.title()}: {region_type} {region_number} successfully unblurred.")
            else:
                messagebox.showerror("Error", "Original region data not found.")
                print(f"Unblur {region_type.title()}: Original region data not found for {region_number}.")
        else:
            messagebox.showerror("Error", "Decryption succeeded, but the message does not match the expected value. Incorrect attributes.")
            print(f"Unblur {region_type.title()}: Decryption succeeded, but hashes do not match.")

    def mark_region_unblurred(self, region_type, region_number):
        # Remove the ciphertext and original region to prevent future blurring
        try:
            self.ciphertext_map[region_type].pop(region_number - 1)
            self.original_regions[region_type].pop(region_number)
            print(f"Region {region_number} of type {region_type} marked as unblurred.")
        except IndexError:
            print(f"Failed to mark region {region_number} of type {region_type} as unblurred.")

# ------------------------------------- Main ------------------------------------- #

def main():
    root = tk.Tk()
    app = VideoBlurringTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()
