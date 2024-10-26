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
policy = 'attributeA'  # Policy for unblurring
hash_map = {}
ciphertext_map = {}
video_capture = None
number_plate_region = None
blur_enabled = True  # Flag for toggling blur

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

# Tkinter GUI Functions
def select_video():
    global video_capture, ciphertext_map, hash_map, number_plate_region

    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if video_path:
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            messagebox.showerror("Error", "Cannot open the selected video.")
            return

        # Load the pre-trained Haar Cascade classifier for number plate detection
        cascade_path = 'haarcascade_russian_plate_number.xml'  # Ensure this path is correct
        if not os.path.exists(cascade_path):
            messagebox.showerror("Error", f"Haar Cascade file not found at {cascade_path}.")
            return

        number_plate_cascade = cv2.CascadeClassifier(cascade_path)

        # Initialize the video frame for hashing and encryption
        ret, frame = video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read the video.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = number_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))

        if len(plates) == 0:
            messagebox.showerror("Error", "No number plate detected in the first frame.")
            return

        # For simplicity, consider the first detected number plate
        (x, y, w, h) = plates[0]
        number_plate_region = (x, y, x + w, y + h)

        original_plate_region = frame[y:y+h, x:x+w]
        hash_value = group.hash(original_plate_region.tobytes(), ZR)
        M = pair(pk['g'], pk['g2']) ** hash_value
        hash_map[1] = M  # Store hash for verification

        # Encrypt the number plate hash with the defined policy
        ciphertext = cpabe.encrypt(pk, M, policy)
        ciphertext_map[1] = ciphertext

        # Reset the video to start
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Start the video playback in a separate thread
        Thread(target=play_video, daemon=True).start()

def play_video():
    global blur_enabled, video_capture, number_plate_region

    number_plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = number_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))

        # Update number_plate_region if detection changes
        if len(plates) > 0:
            (x, y, w, h) = plates[0]
            number_plate_region = (x, y, x + w, y + h)
            if blur_enabled:
                # Blur the detected number plate
                plate_region = frame[y:y+h, x:x+w]
                blurred_plate = cv2.GaussianBlur(plate_region, (99, 99), 30)
                frame[y:y+h, x:x+w] = blurred_plate
        else:
            # If no plate detected, optionally handle it
            pass

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

def toggle_blur():
    global blur_enabled
    blur_enabled = not blur_enabled

def unblur_number_plate():
    global pk, mk, hash_map, ciphertext_map, blur_enabled

    if not number_plate_region:
        messagebox.showerror("Error", "No video selected or number plate not detected.")
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
        ciphertext = ciphertext_map.get(1, None)
        if not ciphertext:
            messagebox.showerror("Error", "Ciphertext not found.")
            return

        rec_msg = cpabe.decrypt(pk, sk, ciphertext)
    except Exception as e:
        messagebox.showerror("Error", f"Decryption failed:\n{str(e)}")
        return

    # Verify the decrypted message against the stored hash
    expected_msg = hash_map.get(1, None)
    if rec_msg and expected_msg and group.serialize(rec_msg) == group.serialize(expected_msg):
        blur_enabled = False  # Disable blurring
        messagebox.showinfo("Success", "Number plate successfully unblurred.")
    else:
        messagebox.showerror("Error", "Incorrect attributes or decryption failed.")

def main():
    initialize_cpabe()

    # Tkinter setup
    root = tk.Tk()
    root.title("Video Number Plate Blurring Tool")
    root.geometry("1000x700")

    btn_select = tk.Button(root, text="Select Video", command=select_video)
    btn_select.pack(pady=10)

    global label_video
    label_video = tk.Label(root)
    label_video.pack(side=tk.TOP, pady=10)

    btn_unblur = tk.Button(root, text="Unblur Number Plate", command=unblur_number_plate)
    btn_unblur.pack(pady=10)

    root.mainloop()

    # Release resources after the window is closed
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
