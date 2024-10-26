import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from threading import Thread

# Initialize Tkinter
root = tk.Tk()
root.title("Video with Toggleable Blur")

# Load video file
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

# Check if a file was selected
if not video_path:
    tk.messagebox.showerror("Error", "No video file selected.")
    root.destroy()
    exit()

# Open the selected video file
cap = cv2.VideoCapture(video_path)

# Load the pre-trained Haar Cascade classifier for number plate detection
number_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Initialize a flag to toggle blurring
blur_enabled = True

# Function to toggle the blur status
def toggle_blur():
    global blur_enabled
    blur_enabled = not blur_enabled

# Set up the GUI layout
canvas = tk.Canvas(root, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
canvas.pack()

# Add the toggle button below the video
unblur_button = tk.Button(root, text="Toggle Unblur", command=toggle_blur)
unblur_button.pack()

# Function to update the video frame
def update_frame():
    global blur_enabled
    if cap.isOpened():
        ret, frame = cap.read()

        # Reset the video if it reaches the end
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        # Convert the frame to grayscale (required by the detector)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect number plates in the frame
        plates = number_plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 20)
        )

        # Loop over each detected number plate and apply blur if enabled
        for (x, y, w, h) in plates:
            if blur_enabled:
                # Apply blur
                plate_region = frame[y:y+h, x:x+w]
                for _ in range(3):  # Apply multiple blur passes for stronger effect
                    plate_region = cv2.GaussianBlur(plate_region, (99, 99), 30)
                # Further enhance by resizing (downsampling and then upsampling)
                small = cv2.resize(plate_region, (0, 0), fx=0.1, fy=0.1)
                plate_region = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                # Place the blurred region back into the frame
                frame[y:y+h, x:x+w] = plate_region

        # Convert the frame to ImageTk format for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

    # Schedule the next frame update
    root.after(10, update_frame)

# Start the video loop
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Release resources after the window is closed
cap.release()
