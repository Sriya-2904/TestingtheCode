import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageFilter, ImageEnhance # Used for image processing
import cv2 # OpenCV for video, object detection, and image manipulation
import numpy as np # For numerical operations with image arrays
import pytesseract # For OCR

# --- Configuration ---
# IMPORTANT: Adjust this path if Tesseract is not found automatically
# On Windows, it might be 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# On macOS/Linux, it's usually in PATH, so '' or 'tesseract' is fine.
pytesseract.pytesseract.tesseract_cmd = r'' # <--- ADJUST THIS IF TESSERACT IS NOT FOUND

# Paths to Haar Cascade XML files (MUST BE IN THE SAME DIRECTORY AS THIS SCRIPT)
CAR_CASCADE_PATH = 'haarcascade_car.xml'
PLATE_CASCADE_PATH = 'haarcascade_russian_plate_number.xml'

# --- Image Enhancement & OCR Functions ---

def preprocess_plate_for_ocr(plate_img):
    """
    Applies a series of image processing steps to a detected number plate
    to improve its readability for OCR.
    """
    if plate_img is None:
        return None

    # Convert to grayscale
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Denoising (Non-Local Means) - crucial for blurry/noisy plates
    # h=10 is a good starting point, increase for more noise.
    denoised_plate = cv2.fastNlMeansDenoising(gray_plate, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Adaptive Thresholding - to make text pure black/white against background
    # blockSize: neighborhood size (odd). C: constant subtracted.
    # Experiment with blockSize (e.g., 11, 15, 21) and C (e.g., 2, 5, 10)
    thresh_plate = cv2.adaptiveThreshold(
        denoised_plate, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, # Keep this odd. Higher values for more uniform backgrounds.
        C=2          # Subtracts from mean. Adjust if text is too thick/thin.
    )

    # Invert colors if text is white on black (Tesseract often prefers dark text on light background)
    # Check if the majority of the image is dark (background is darker than text)
    if np.mean(thresh_plate) < 127: # If average pixel value is low (dark)
        thresh_plate = cv2.bitwise_not(thresh_plate)

    # Optional: Enhance contrast and sharpness using PIL if needed after thresholding
    pil_img = Image.fromarray(thresh_plate)
    enhancer_contrast = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer_contrast.enhance(1.2) # Slight contrast boost
    enhancer_sharpness = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer_sharpness.enhance(1.5) # Slight sharpness boost

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) # Convert back to OpenCV format

def get_blur_value(image):
    """Calculates the variance of Laplacian to estimate image blur."""
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return 0
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_text_from_plate(plate_img):
    """Extracts text from a preprocessed number plate image using Tesseract OCR."""
    if plate_img is None:
        return "No Plate Image"

    # Convert to PIL Image for Tesseract
    pil_plate = Image.fromarray(plate_img)

    # Use Tesseract to do OCR
    # config: '--psm 8' is for a single word/line, often good for number plates.
    # --oem 3 is for default OCR engine modes.
    # -l eng: specifies English language. If your plates are in another language, change 'eng'.
    try:
        text = pytesseract.image_to_string(pil_plate, config='--psm 8 --oem 3 -l eng')
        # Clean up the text: remove non-alphanumeric, remove whitespace, convert to uppercase
        cleaned_text = "".join(filter(str.isalnum, text)).upper()
        return cleaned_text if cleaned_text else "No Text Found"
    except pytesseract.TesseractNotFoundError:
        messagebox.showerror("Tesseract Error",
                             "Tesseract OCR engine not found. "
                             "Please install Tesseract from https://tesseract-ocr.github.io/tessdoc/Downloads.html "
                             "and ensure it's added to your system's PATH, or update the tesseract_cmd path in the script.")
        return "Tesseract Not Installed"
    except Exception as e:
        return f"OCR Error: {e}"

# --- Main Video Processing Logic ---

def process_video_for_plates(video_path, output_dir):
    """
    Detects vehicles and number plates in a video, selects the best quality plate,
    enhances it, and extracts text.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save output images.

    Returns:
        tuple: (list of detected plate texts, path to saved vehicle image, path to saved plate image)
    """
    if not os.path.exists(CAR_CASCADE_PATH):
        messagebox.showerror("Error", f"Car cascade '{CAR_CASCADE_PATH}' not found. Please download and place it in the script's directory.")
        return [], None, None
    if not os.path.exists(PLATE_CASCADE_PATH):
        messagebox.showerror("Error", f"Plate cascade '{PLATE_CASCADE_PATH}' not found. Please download and place it in the script's directory.")
        return [], None, None

    car_cascade = cv2.CascadeClassifier(CAR_CASCADE_PATH)
    plate_cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video file: {video_path}")
        return [], None, None

    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    detected_texts = []
    best_plate_image = None
    best_plate_blur = -1
    best_vehicle_frame = None

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        # Process every Nth frame to speed up
        if frame_count % 5 != 0: # Process every 5th frame
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detect vehicles
        # minSize: minimum possible object size. adjust based on how far vehicles are.
        # scaleFactor: how much the image size is reduced at each image scale.
        cars = car_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x_car, y_car, w_car, h_car) in cars:
            # Draw rectangle around car (for visualization, optional)
            # cv2.rectangle(frame, (x_car, y_car), (x_car + w_car, y_car + h_car), (0, 255, 0), 2)

            # Define region of interest for number plate detection (within the car)
            car_roi_gray = gray_frame[y_car:y_car + h_car, x_car:x_car + w_car]
            car_roi_color = frame[y_car:y_car + h_car, x_car:x_car + w_car]

            # 2. Detect number plates within the car ROI
            plates = plate_cascade.detectMultiScale(car_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

            for (x_plate, y_plate, w_plate, h_plate) in plates:
                # Ensure plate dimensions are reasonable (e.g., aspect ratio)
                aspect_ratio = w_plate / h_plate
                if 2.5 < aspect_ratio < 6.0: # Typical aspect ratio for number plates (e.g., 4:1)
                    plate_roi = car_roi_color[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]

                    # 3. Assess quality (blur)
                    current_blur = get_blur_value(plate_roi)

                    # If this plate is less blurry than the best found so far, update
                    if current_blur > best_plate_blur: # Higher Laplacian variance means less blur
                        best_plate_blur = current_blur
                        best_plate_image = plate_roi.copy() # Make a copy
                        best_vehicle_frame = frame.copy() # Save the full frame with the vehicle

    cap.release() # Release the video capture object

    saved_vehicle_path = None
    saved_plate_path = None
    extracted_text = "No plate detected or text extracted."

    if best_plate_image is not None:
        # Save the best vehicle frame
        video_base_name = os.path.splitext(os.path.basename(video_path))[0]
        saved_vehicle_path = os.path.join(output_dir, f"{video_base_name}_detected_vehicle.png")
        cv2.imwrite(saved_vehicle_path, best_vehicle_frame)

        # Enhance and save the best number plate image
        processed_plate_image = preprocess_plate_for_ocr(best_plate_image)
        if processed_plate_image is not None:
            saved_plate_path = os.path.join(output_dir, f"{video_base_name}_detected_number_plate.png")
            cv2.imwrite(saved_plate_path, processed_plate_image)

            # Extract text from the processed plate
            extracted_text = extract_text_from_plate(processed_plate_image)
            detected_texts.append(extracted_text)
        else:
            extracted_text = "Plate enhancement failed."
    else:
        messagebox.showinfo("No Detections", "No clear number plate was detected in the video.")


    return detected_texts, saved_vehicle_path, saved_plate_path

# --- GUI Application Class ---

class VideoProcessorApp:
    def __init__(self, master):
        self.master = master
        master.title("Vehicle and Number Plate OCR")
        master.geometry("600x400")
        master.resizable(False, False)

        self.video_file_path = ""
        self.output_dir = os.path.join(os.getcwd(), "video_ocr_output")
        os.makedirs(self.output_dir, exist_ok=True)

        tk.Label(master, text="Selected Video File:").pack(pady=10)
        self.video_path_display = tk.Entry(master, width=70, state="readonly")
        self.video_path_display.pack()

        self.browse_button = tk.Button(master, text="Browse Video", command=self.browse_video)
        self.browse_button.pack(pady=5)

        self.process_button = tk.Button(master, text="Process Video & Extract Plate", command=self.process_video)
        self.process_button.pack(pady=10)
        self.process_button.config(state=tk.DISABLED)

        tk.Label(master, text="Extracted Text:").pack(pady=10)
        self.extracted_text_display = tk.Text(master, height=5, width=60, state="disabled", wrap=tk.WORD)
        self.extracted_text_display.pack()

        self.status_label = tk.Label(master, text="Ready: Select a video file.")
        self.status_label.pack(pady=10)

        # Initial check for cascade files
        self.check_cascade_files()

    def check_cascade_files(self):
        """Checks if Haar cascade files are present."""
        car_exists = os.path.exists(CAR_CASCADE_PATH)
        plate_exists = os.path.exists(PLATE_CASCADE_PATH)

        if not car_exists or not plate_exists:
            messagebox.showerror("Missing Files",
                                 f"Required cascade files missing:\n"
                                 f"Car cascade ({CAR_CASCADE_PATH}): {car_exists}\n"
                                 f"Plate cascade ({PLATE_CASCADE_PATH}): {plate_exists}\n\n"
                                 "Please download them from OpenCV's GitHub and place them in the script's directory.")
            self.process_button.config(state=tk.DISABLED)
            self.status_label.config(text="ERROR: Missing cascade XML files. Cannot process.")
            return False
        return True

    def browse_video(self):
        """Opens a file dialog for the user to select a video file."""
        f_types = [('Video Files', '*.mp4 *.avi *.mov *.mkv')]
        path = filedialog.askopenfilename(filetypes=f_types)
        if path:
            self.video_file_path = path
            self.video_path_display.config(state="normal")
            self.video_path_display.delete(0, tk.END)
            self.video_path_display.insert(0, self.video_file_path)
            self.video_path_display.config(state="readonly")
            self.status_label.config(text=f"Video selected: {os.path.basename(self.video_file_path)}")
            if self.check_cascade_files(): # Enable only if cascades are present
                self.process_button.config(state=tk.NORMAL)
            self.extracted_text_display.config(state="normal")
            self.extracted_text_display.delete(1.0, tk.END)
            self.extracted_text_display.config(state="disabled")
        else:
            self.status_label.config(text="No video selected.")
            self.process_button.config(state=tk.DISABLED)
            self.video_file_path = ""

    def process_video(self):
        """Triggers the video processing and OCR."""
        if not self.video_file_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return

        self.status_label.config(text="Processing video... This may take time.")
        self.process_button.config(state=tk.DISABLED)
        self.extracted_text_display.config(state="normal")
        self.extracted_text_display.delete(1.0, tk.END)
        self.extracted_text_display.config(state="disabled")
        self.master.update_idletasks()

        try:
            plate_texts, vehicle_img_path, plate_img_path = process_video_for_plates(self.video_file_path, self.output_dir)

            self.extracted_text_display.config(state="normal")
            if plate_texts:
                self.extracted_text_display.insert(tk.END, "\n".join(plate_texts))
                self.status_label.config(text=f"Processing complete. Output in '{self.output_dir}'")
                messagebox.showinfo("Success",
                                    f"Processing complete!\n"
                                    f"Detected Text(s): {', '.join(plate_texts) if plate_texts else 'N/A'}\n"
                                    f"Vehicle image saved to: {vehicle_img_path}\n"
                                    f"Plate image saved to: {plate_img_path}\n"
                                    f"Check the '{self.output_dir}' folder.")
            else:
                self.extracted_text_display.insert(tk.END, "No text found.")
                self.status_label.config(text="No text or plate detected.")
                messagebox.showinfo("Result", "No clear number plate or readable text could be extracted.")

            self.extracted_text_display.config(state="disabled")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during video processing: {e}")
            self.status_label.config(text="Error during processing.")
        finally:
            self.process_button.config(state=tk.NORMAL) # Re-enable button

# --- Main execution block for GUI ---

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorApp(root)
    root.mainloop()
