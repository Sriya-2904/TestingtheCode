import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageFilter, ImageEnhance # Import Image, ImageFilter, ImageEnhance for specific operations
import cv2 # OpenCV for advanced image processing, especially text-focused
import numpy as np # NumPy for array operations with OpenCV

# --- Image Enhancement Function for Text Readability ---

def enhance_image_for_text_readability(image_path, output_path_prefix):
    """
    Enhances an image specifically to improve readability of text and numbers,
    using a pipeline of OpenCV and Pillow operations.
    This diagnostic version saves intermediate steps for debugging.

    Args:
        image_path (str): The file path to the input image.
        output_path_prefix (str): A prefix for output paths (e.g., 'path/to/output/my_image').
                                  Intermediate files will be saved as 'my_image_grayscale.png', etc.

    Returns:
        str or None: The path to the final saved image if successful, None otherwise.
    """
    try:
        # Define output folder for diagnostics
        output_dir = os.path.dirname(output_path_prefix)
        base_name = os.path.basename(output_path_prefix)

        # 1. Load image using OpenCV
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError(f"OpenCV could not load image from '{image_path}'. Check path or file corruption.")

        # Save original color image as a reference
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), img_cv)

        # Convert to grayscale: Essential for text processing to remove color noise
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_grayscale.png"), gray_img) # Diagnostic save

        # 2. Denoising: Apply a Non-Local Means Denoising specifically for grayscale
        # h: filter strength. Adjust this value (e.g., 5 to 20) based on noise level.
        denoised_gray = cv2.fastNlMeansDenoising(gray_img, None, h=15, templateWindowSize=7, searchWindowSize=21)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_denoised.png"), denoised_gray) # Diagnostic save

        # 3. Adaptive Thresholding: Crucial for making text clear even with uneven lighting
        # This converts the image to pure black and white, making text highly contrasted.
        # block_size: Size of a pixel neighborhood. Must be odd. Try 11, 15, 21.
        # C: Constant subtracted from the mean. Try 2, 5, 10.
        thresholded_img = cv2.adaptiveThreshold(
            denoised_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size=11, # <--- IMPORTANT: TRY ADJUSTING THIS (e.g., 15, 21)
            C=2 # <--- IMPORTANT: TRY ADJUSTING THIS (e.g., 5, 10)
        )

        # Invert colors if text is white on black (common for thresholded images)
        # Check top-left corner for approximate background brightness. If dark, invert.
        # This assumes text is generally a smaller portion of the image than background.
        if np.mean(thresholded_img[0:min(50, thresholded_img.shape[0]), 0:min(50, thresholded_img.shape[1])]) < 127:
             thresholded_img = cv2.bitwise_not(thresholded_img)

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_thresholded.png"), thresholded_img) # Diagnostic save

        # Convert back to PIL Image to apply Pillow's sharpness and contrast (sometimes better after thresholding)
        pil_img = Image.fromarray(thresholded_img)

        # 4. Enhance Sharpness (Pillow): Apply a final sharpness boost
        enhancer_sharp = ImageEnhance.Sharpness(pil_img)
        sharpened_pil = enhancer_sharp.enhance(1.5) # Factor 1.0 is original, 1.5-2.0 is usually good
        sharpened_pil.save(os.path.join(output_dir, f"{base_name}_sharpened_final_step.png")) # Diagnostic save

        # 5. Enhance Contrast (Pillow): Final contrast boost
        enhancer_contrast = ImageEnhance.Contrast(sharpened_pil)
        final_pil_img = enhancer_contrast.enhance(1.2) # Factor 1.0 is original, 1.2-1.5 is usually good

        # Save the final enhanced image. Always save as PNG for text quality.
        final_output_path = os.path.join(output_dir, f"{base_name}_final_enhanced_text.png")
        final_pil_img.save(final_output_path)
        return final_output_path

    except FileNotFoundError:
        messagebox.showerror("Error", f"Input file '{image_path}' not found.")
        return None
    except ValueError as ve:
        messagebox.showerror("Value Error", f"Image processing error: {ve}")
        return None
    except Exception as e:
        messagebox.showerror("Enhancement Error", f"Failed to enhance image: {e}")
        return None

# --- GUI Application Class ---

class ImageEnhancerApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Enhancer for Text Readability (Diagnostic)")
        master.geometry("500x250")
        master.resizable(False, False)

        self.input_file_path = ""
        # New output folder name specific to this text-focused enhancement
        self.output_dir = os.path.join(os.getcwd(), "enhanced_images_for_text_diagnostic")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Labels and Entry for Input File
        self.label_input = tk.Label(master, text="Selected Image:")
        self.label_input.pack(pady=10)

        self.input_path_display = tk.Entry(master, width=60, state="readonly")
        self.input_path_display.pack()

        # Browse Button
        self.browse_button = tk.Button(master, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=5)

        # Single Enhance Button for Text Readability
        self.enhance_button = tk.Button(master, text="Enhance Text Readability", command=self.perform_enhancement)
        self.enhance_button.pack(pady=10)
        self.enhance_button.config(state=tk.DISABLED) # Disable until image is selected

        # Status Label
        self.status_label = tk.Label(master, text="Ready: Browse an image to begin.")
        self.status_label.pack(pady=10)

    def browse_image(self):
        """Opens a file dialog for the user to select an image."""
        f_types = [('Image Files', '*.jpeg *.jpg *.png *.bmp *.gif')]
        path = filedialog.askopenfilename(filetypes=f_types)
        if path:
            self.input_file_path = path
            self.input_path_display.config(state="normal")
            self.input_path_display.delete(0, tk.END)
            self.input_path_display.insert(0, self.input_file_path)
            self.input_path_display.config(state="readonly")
            self.status_label.config(text=f"Image selected: {os.path.basename(self.input_file_path)}")
            self.enhance_button.config(state=tk.NORMAL) # Enable enhance button
        else:
            self.status_label.config(text="No image selected.")
            self.enhance_button.config(state=tk.DISABLED)
            self.input_file_path = ""

    def perform_enhancement(self):
        """
        Executes the text readability enhancement process when the Enhance button is clicked.
        """
        if not self.input_file_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        base_name = os.path.basename(self.input_file_path)
        name, ext = os.path.splitext(base_name)
        # Use a base name for the output that includes original name
        output_path_prefix = os.path.join(self.output_dir, name) # No extension here

        self.status_label.config(text="Enhancing image for text readability... Please wait.")
        self.master.update_idletasks() # Update GUI to show message

        # Disable button during processing to prevent multiple clicks
        self.enhance_button.config(state=tk.DISABLED)

        try:
            final_result_path = enhance_image_for_text_readability(self.input_file_path, output_path_prefix)
            if final_result_path:
                messagebox.showinfo("Success",
                                    f"Image enhanced for text readability.\n"
                                    f"Check the '{self.output_dir}' folder for the final image and intermediate steps.")
                self.status_label.config(text=f"Text enhancement complete. Files saved in: {self.output_dir}")
            else:
                self.status_label.config(text="Text enhancement failed.")
        finally:
            # Re-enable button after processing (even if it failed)
            self.enhance_button.config(state=tk.NORMAL)

# --- Main execution block for GUI ---

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancerApp(root)
    root.mainloop()
