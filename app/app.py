import sys
import os
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.append(os.path.join(os.path.dirname(__file__), '../ML'))
import inference as inf 

sys.path.append(os.path.join(os.path.dirname(__file__), '../pathing'))
import pathing

class ClimbingPathGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Beta than Ever!")
        self.root.geometry("1600x900")

        # Title Label
        self.title_label = tk.Label(self.root, text="Beta than Ever!", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # Frame for the buttons and class selection
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(pady=10)

        # Upload Button
        self.upload_button = tk.Button(
            self.controls_frame,
            text="Upload Problem",
            command=self.upload_image,
            font=("Arial", 14)
        )
        self.upload_button.pack(side=tk.LEFT, padx=10)

        # Frame for class selection and submit button
        self.class_frame = tk.Frame(self.controls_frame)
        self.class_frame.pack(side=tk.LEFT, padx=10)

        # Label for dropdown
        self.class_label = tk.Label(
            self.class_frame,
            text="Select Hold Class:",
            font=("Arial", 12)
        )
        self.class_label.pack(side=tk.LEFT, padx=5)
                
        # Class dropdown menu
        self.classes = tk.StringVar()
        self.class_dropdown = ttk.Combobox(self.class_frame, width = 27, textvariable=self.classes)
        self.class_dropdown['values'] =(
            'Black', 'Blue', 'Brown', 'Cream', 'Gray', 'Green',
            'Orange', 'Pink', 'Purple', 'Red', 'White',
            'Yellow'
        )
        self.class_dropdown.current(7)
        self.class_dropdown.pack(side=tk.LEFT, padx=5)

        # Submit Button
        self.submit_button = tk.Button(
            self.class_frame,
            text="Submit",
            command=self.submit_image,
            font=("Arial", 14)
        )
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        # User info Height and Foothold Frame
        self.user_frame = tk.Frame(self.controls_frame)
        self.user_frame.pack(side=tk.LEFT, padx=5)
        
        self.foot_label = tk.Label(
            self.user_frame,
            text="Enter Foothold ID",
            font=("Arial, 12")
        )
        self.foot_label.pack(side=tk.LEFT, padx=5)
        self.foot_id_entry = tk.Entry(self.user_frame)
        self.foot_id_entry.pack(side=tk.LEFT, padx=5)
        
        self.height_label = tk.Label(
            self.user_frame,
            text="Enter Height",
            font=("Arial, 12")
        )
        self.height_label.pack(side=tk.LEFT, padx=5)
        self.height_entry = tk.Entry(self.user_frame, width=6)
        self.height_entry.pack(side=tk.LEFT, padx=5)
        
        self.LH_label = tk.Label(
            self.user_frame,
            text="LH",
            font=("Arial, 12")
        )
        self.LH_label.pack(side=tk.LEFT, padx=5)
        self.LH_entry = tk.Entry(self.user_frame, width=3)
        self.LH_entry.pack(side=tk.LEFT, padx=5)
        
        self.RH_label = tk.Label(
            self.user_frame,
            text="RH",
            font=("Arial, 12")
        )
        self.RH_label.pack(side=tk.LEFT, padx=5)
        self.RH_entry = tk.Entry(self.user_frame, width=3)
        self.RH_entry.pack(side=tk.LEFT, padx=5)
        
        self.LF_label = tk.Label(
            self.user_frame,
            text="LF",
            font=("Arial, 12")
        )
        self.LF_label.pack(side=tk.LEFT, padx=5)
        self.LF_entry = tk.Entry(self.user_frame, width=3)
        self.LF_entry.pack(side=tk.LEFT, padx=5)
        
        self.RF_label = tk.Label(
            self.user_frame,
            text="RF",
            font=("Arial, 12")
        )
        self.RF_label.pack(side=tk.LEFT, padx=5)
        self.RF_entry = tk.Entry(self.user_frame, width=3)
        self.RF_entry.pack(side=tk.LEFT, padx=5)
        
        
        # Generate Steps Button
        self.upload_button = tk.Button(
            self.controls_frame,
            text="Generate Steps",
            command=self.generate_steps,
            font=("Arial", 14)
        )
        self.upload_button.pack(side=tk.LEFT, padx=10)

        # Frame for the image and text
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # Image Canvas
        self.processed_canvas = tk.Canvas(
            self.display_frame,
            bg='gray',
            width=800,
            height=750
        )
        self.processed_canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Text Field
        self.text_frame = tk.Frame(self.display_frame)
        self.text_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.text_label = tk.Label(
            self.text_frame,
            text="Steps:",
            font=("Arial", 14)
        )
        self.text_label.pack(anchor=tk.NW)

        self.text_field = tk.Text(
            self.text_frame,
            wrap=tk.WORD,
            font=("Arial", 12)
        )
        self.text_field.pack(fill=tk.BOTH, expand=True)

        self.processed_photo_image = None
        self.processed_image_np = None
        self.image_path = None
        self.holds = None
        self.target_class = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
        )
        if not file_path:
            messagebox.showinfo("Error!", f"{file_path} is not a file path")
        else:
            self.image_path = file_path

    def submit_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image uploaded. Please upload an image first.")
            return

        self.target_class = self.class_dropdown.get()

        try:
            processed_image = inf.generate_image(self.image_path, self.target_class)
            if processed_image is not None:
                self.processed_image_np = processed_image
                
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(processed_image_rgb)
                pil_image = self.resize_image_with_aspect_ratio(pil_image, 800, 750)
                self.processed_photo_image = ImageTk.PhotoImage(pil_image)

                self.processed_canvas.delete("all")
                self.processed_canvas.create_image(400, 375, image=self.processed_photo_image, anchor=tk.CENTER)
                self.holds = inf.filter_holds(inf.detect_holds(self.image_path), self.target_class)
            else:
                messagebox.showerror("Error", "Failed to process the image.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

    def resize_image_with_aspect_ratio(self, pil_image, max_width, max_height):
        original_width, original_height = pil_image.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        return pil_image.resize(new_size)
    
    def generate_steps(self):
        start_state = {
            "right_hand": int(self.RH_entry.get()),
            "left_hand": int(self.LH_entry.get()),
            "right_foot": int(self.RF_entry.get()),
            "left_foot": int(self.LF_entry.get())
        }
        
        steps = pathing.path(self.holds, self.height_entry.get(), self.foot_id_entry.get(), self.target_class, start_state)
        
        if steps:
            self.text_field.delete("1.0", tk.END)  
            count = 0
            for step in steps:
                self.text_field.insert(
                    tk.END,
                    'Step ' + str(count) + ': ' + step + '\n'
                )
                count += 1
        else:
            self.text_field.delete("1.0", tk.END)
            self.text_field.insert(tk.END, "Unable to determine steps")


def main():
    root = tk.Tk()
    app = ClimbingPathGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
