import cv2
from tkinter import Tk, filedialog, messagebox, simpledialog, Button
from ultralytics import YOLO
import torch
import threading
from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, G2, pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.ABEnc import ABEnc

# Initialize pairing group for CP-ABE
group = PairingGroup('SS512')
cpabe = None
pk = None
mk = None
policy_plate = 'attributeA'  # Policy for unblurring number plates
policy_face = 'attributeB'   # Policy for unblurring faces
hash_map = {}
ciphertext_map = {}

# Global variables to control blur/unblur toggle
blur_faces = True
blur_plates = True

# Initialize CP-ABE scheme
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

# Initialize CP-ABE
def initialize_cpabe():
    global cpabe, pk, mk
    cpabe = CPabe_BSW07(group)
    (pk, mk) = cpabe.setup()

initialize_cpabe()

def select_and_process_video():
    global blur_faces, blur_plates, pk, mk, hash_map, ciphertext_map  # Declare global variables at the start of the function
    
    Tk().withdraw()  # Hide the root tkinter window
    video_path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if not video_path:
        messagebox.showinfo("No Selection", "No video selected.")
        return

    # Paths to the models
    face_model_path = "/home/aditya/MTP3/yolov8n-face-lindevs.pt"  # Replace with your face detection model path
    plate_model_path = "/home/aditya/MTP3/best.pt"  # Replace with your number plate detection model path

    try:
        # Load the YOLO models
        face_model = YOLO(face_model_path)
        plate_model = YOLO(plate_model_path)

        # Move models to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        face_model.to(device)
        plate_model.to(device)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load YOLO models: {e}")
        return

    # Open the selected video
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        messagebox.showerror("Error", "Cannot open the selected video.")
        return

    frame_skip = 2  # Skip every 2nd frame for faster processing
    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        frame_count += 1
        if not ret or frame_count % frame_skip != 0:
            continue

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (1280, 720))

        # Detect faces
        face_results = face_model(frame_resized)
        for face_box in face_results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, face_box[:4])
            face_region = frame_resized[y1:y2, x1:x2]
            hash_value = group.hash(face_region.tobytes(), ZR)
            M = pair(pk['g'], pk['g2']) ** hash_value
            ciphertext_map['face'] = cpabe.encrypt(pk, M, policy_face)
            hash_map['face'] = M
            if blur_faces:
                # Blur the detected face region
                face_region = cv2.GaussianBlur(face_region, (51, 51), 0)
                frame_resized[y1:y2, x1:x2] = face_region
            else:
                # Draw rectangle for face
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for faces
                cv2.putText(frame_resized, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detect number plates
        plate_results = plate_model(frame_resized)
        for plate_box in plate_results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, plate_box[:4])
            plate_region = frame_resized[y1:y2, x1:x2]
            hash_value = group.hash(plate_region.tobytes(), ZR)
            M = pair(pk['g'], pk['g2']) ** hash_value
            ciphertext_map['plate'] = cpabe.encrypt(pk, M, policy_plate)
            hash_map['plate'] = M
            if blur_plates:
                # Blur the detected number plate region
                plate_region = cv2.GaussianBlur(plate_region, (51, 51), 0)
                frame_resized[y1:y2, x1:x2] = plate_region
            else:
                # Draw rectangle for number plate
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for plates
                cv2.putText(frame_resized, "Plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detections
        cv2.imshow("Face and Plate Detection", frame_resized)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

def unblur_region(region_key, policy):
    global pk, mk, hash_map, ciphertext_map, blur_faces, blur_plates

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
        if region_key == 'face':
            blur_faces = False
        elif region_key == 'plate':
            blur_plates = False
        messagebox.showinfo("Success", f"{region_key.replace('_', ' ').capitalize()} successfully unblurred.")
    else:
        messagebox.showerror("Error", "Incorrect attributes or decryption failed.")

def unblur_number_plate():
    unblur_region('plate', policy_plate)

def unblur_face():
    unblur_region('face', policy_face)

def start_video_processing():
    threading.Thread(target=select_and_process_video).start()

if __name__ == "__main__":
    root = Tk()
    root.title("Video Processing Tool")
    root.protocol("WM_DELETE_WINDOW", root.quit)

    select_button = Button(root, text="Select and Process Video", command=start_video_processing)
    select_button.pack(pady=10)

    unblur_plate_button = Button(root, text="Unblur Number Plate", command=unblur_number_plate)
    unblur_plate_button.pack(pady=10)

    unblur_face_button = Button(root, text="Unblur Face", command=unblur_face)
    unblur_face_button.pack(pady=10)

    quit_button = Button(root, text="Quit", command=root.quit)
    quit_button.pack(pady=10)

    root.mainloop()
