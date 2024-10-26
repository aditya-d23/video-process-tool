import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Initialize Tkinter and hide the root window
root = tk.Tk()
root.withdraw()

# Open file dialog to select video file
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

# Check if a file was selected
if not video_path:
    messagebox.showerror("Error", "No video file selected.")
    exit()

# Open the selected video file
cap = cv2.VideoCapture(video_path)

# Load the pre-trained Haar Cascade classifier for number plate detection
number_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (required by the detector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect number plates in the frame with adjusted parameters
    plates = number_plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Adjust scaleFactor to improve detection sensitivity
        minNeighbors=4,       # Adjust minNeighbors to control false positives
        minSize=(60, 20)      # Adjust minSize based on the size of plates in the video
    )

    # Loop over each detected number plate and apply an enhanced blur
    for (x, y, w, h) in plates:
        # Extract the region of interest (number plate area)
        plate_region = frame[y:y+h, x:x+w]
        
        # Apply a strong blur by resizing and applying Gaussian blur multiple times
        for _ in range(3):  # Apply multiple blur passes for stronger effect
            plate_region = cv2.GaussianBlur(plate_region, (99, 99), 30)
        
        # Optional: further enhance by resizing (downsampling and then upsampling)
        small = cv2.resize(plate_region, (0, 0), fx=0.1, fy=0.1)
        plate_region = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Place the blurred region back into the frame
        frame[y:y+h, x:x+w] = plate_region

    # Display the frame with blurred number plates
    cv2.imshow("Video with Enhanced Blurred Number Plates", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
