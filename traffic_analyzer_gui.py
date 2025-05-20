import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

OBJECT_DETECTOR_PATH = 'saved_models/custom_object_detector.keras'
COLOR_CLASSIFIER_PATH = 'saved_models/custom_color_classifier.keras'

OBJECT_CLASSES = {1: 'car', 2: 'person'} 
CAR_CLASS_INDEX = 1
PERSON_CLASS_INDEX = 2

try:
    COLOR_CLASSES = sorted([d for d in os.listdir('data/color_classification/train') if os.path.isdir(os.path.join('data/color_classification/train', d))])
    if not COLOR_CLASSES: raise FileNotFoundError
except FileNotFoundError:
    COLOR_CLASSES = ['Black', 'Blue', 'Other', 'Red', 'Silver', 'White']
    print("Warning: Using default COLOR_CLASSES list.")
print(f"Using color classes: {COLOR_CLASSES}")

BOX_COLOR_BLUE_CAR = (0, 0, 255) 
BOX_COLOR_OTHER_CAR = (255, 0, 0) 
BOX_COLOR_PERSON = (0, 255, 0) 
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1

COLOR_IMG_WIDTH = 64
COLOR_IMG_HEIGHT = 64
COLOR_CHANNELS = 3

models = {}
object_detector = None
color_classifier = None
models_loaded = False

try:
    print("Loading models...")
    if os.path.exists(OBJECT_DETECTOR_PATH):
        print("--- Placeholder: Loading Object Detector Skipped ---")
        print("--- Needs actual model loading code ---")
        _
        models['object_detector'] = None 
    else:
        print(f"Warning: Object detector model not found at {OBJECT_DETECTOR_PATH}")
        models['object_detector'] = None

    if os.path.exists(COLOR_CLASSIFIER_PATH):
        models['color_classifier'] = tf.keras.models.load_model(COLOR_CLASSIFIER_PATH)
        print("Color classifier loaded.")
    else:
         print(f"Error: Color classifier model not found at {COLOR_CLASSIFIER_PATH}")
         models['color_classifier'] = None

    if models.get('color_classifier'): 
        models_loaded = True 
        print("Essential color model loaded.")
    if models.get('object_detector'):
         print("Placeholder: Object detector would be loaded here.")

except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False


def predict_objects_placeholder(model, image):
    print("--- WARNING: Using Placeholder Object Detection ---")
    h, w, _ = image.shape
    boxes = [ [int(w*0.1), int(h*0.4), int(w*0.4), int(h*0.7)], 
              [int(w*0.6), int(h*0.5), int(w*0.9), int(h*0.8)], 
              [int(w*0.4), int(h*0.1), int(w*0.6), int(h*0.4)]] 
    scores = [0.98, 0.95, 0.90]
    classes = [CAR_CLASS_INDEX, CAR_CLASS_INDEX, PERSON_CLASS_INDEX] 

    return boxes, scores, classes

def predict_color(model, car_roi):
    if model is None or car_roi is None or car_roi.size == 0:
        return "N/A", 0.0
    try:
        img_resized = cv2.resize(car_roi, (COLOR_IMG_WIDTH, COLOR_IMG_HEIGHT))
        img_normalized = img_resized.astype('float32') / 255.0
        if COLOR_CHANNELS == 1:
             if len(img_normalized.shape) == 3: # Convert if needed
                  img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2GRAY)
             img_normalized = np.expand_dims(img_normalized, axis=-1)

        img_batch = np.expand_dims(img_normalized, axis=0)
        predictions = model.predict(img_batch, verbose=0)
        pred_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        if 0 <= pred_index < len(COLOR_CLASSES):
            return COLOR_CLASSES[pred_index], confidence
        else:
            return "Error", 0.0
    except Exception as e:
        print(f"Color prediction error: {e}")
        return "Error", 0.0

class TrafficAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Color Analyzer")
        self.root.geometry("1000x700")

        self.image_path = None
        self.original_image = None
        self.display_image_tk = None

        self.control_frame = Frame(root, pady=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        self.load_button = Button(self.control_frame, text="Load Image", command=self.load_image, width=15)
        self.load_button.pack(side=tk.LEFT, padx=20, pady=5)
        self.process_button = Button(self.control_frame, text="Analyze Traffic", command=self.process_image, state=tk.DISABLED, width=15)
        self.process_button.pack(side=tk.LEFT, padx=20, pady=5)
        self.info_label = Label(self.control_frame, text="Load an image to analyze.")
        self.info_label.pack(side=tk.LEFT, padx=20)

        self.image_frame = Frame(root, relief=tk.SUNKEN, borderwidth=1)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = Canvas(self.image_frame, bg="lightgrey")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        if not models_loaded:
             messagebox.showerror("Model Load Error", "One or more models failed to load. Analysis may be limited or unavailable.")

    def display_on_canvas(self, image_cv, message):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2:
            canvas_width = 800; canvas_height = 600

        if image_cv is None:
            self.canvas.create_text(canvas_width / 2, canvas_height / 2, text=message, anchor=tk.CENTER, width=canvas_width-20)
            return

        try:
            img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((canvas_width - 20, canvas_height - 20), Image.Resampling.LANCZOS)
            self.display_image_tk = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=self.display_image_tk)
            self.canvas.image = self.display_image_tk
        except Exception as e:
             print(f"Error displaying image: {e}")
             self.canvas.create_text(canvas_width / 2, canvas_height / 2, text="Error Displaying Image", anchor=tk.CENTER)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.image_path = path
        try:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None: raise ValueError("File is not a valid image.")
            self.display_on_canvas(self.original_image, "")
            self.process_button.config(state=tk.NORMAL if models_loaded else tk.DISABLED)
            self.info_label.config(text=f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load image:\n{e}")
            self.image_path = None; self.original_image = None
            self.process_button.config(state=tk.DISABLED)
            self.info_label.config(text="Error loading.")
            self.display_on_canvas(None, "Error loading image")

    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Load an image first.")
            return
        if not models_loaded:
            messagebox.showerror("Model Error", "Cannot process: Models not loaded correctly.")
            return

        self.info_label.config(text="Processing...")
        self.root.update_idletasks()

        processed_img = self.original_image.copy()
        car_count = 0
        person_count = 0

        try:
           
            if models.get('object_detector'):
                 boxes, scores, classes = predict_objects_placeholder(models['object_detector'], processed_img) 
                 print("--- Using ACTUAL (but likely placeholder implementation) Object Detector ---")
            else:
                 boxes, scores, classes = predict_objects_placeholder(None, processed_img) 
                 print("--- Using DUMMY Object Detection due to load failure or placeholder ---")


            detection_threshold = 0.5 

            for i, box in enumerate(boxes):
                if scores[i] < detection_threshold: continue

                x1, y1, x2, y2 = map(int, box)
                class_id = int(classes[i])

                if class_id == CAR_CLASS_INDEX:
                    car_count += 1
                    car_roi = processed_img[y1:y2, x1:x2]
                    color_name, color_conf = predict_color(models.get('color_classifier'), car_roi)

                    box_color = BOX_COLOR_OTHER_CAR
                    label_text = f"Car: {color_name}" 

                    if color_name.lower() == 'blue':
                        box_color = BOX_COLOR_BLUE_CAR

                    cv2.rectangle(processed_img, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(processed_img, label_text, (x1, y1 - 10), FONT, FONT_SCALE, box_color, FONT_THICKNESS, cv2.LINE_AA)

                elif class_id == PERSON_CLASS_INDEX:
                    person_count += 1
                    cv2.rectangle(processed_img, (x1, y1), (x2, y2), BOX_COLOR_PERSON, 2)
                    cv2.putText(processed_img, "Person", (x1, y1 - 10), FONT, FONT_SCALE, BOX_COLOR_PERSON, FONT_THICKNESS, cv2.LINE_AA)

            count_text = f"Cars: {car_count} | People: {person_count}"
            (tw, th), _ = cv2.getTextSize(count_text, FONT, FONT_SCALE*1.1, FONT_THICKNESS)
            cv2.rectangle(processed_img, (5, 5), (10 + tw, 15 + th), (0,0,0), -1)
            cv2.putText(processed_img, count_text, (10, 10 + th), FONT, FONT_SCALE*1.1, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            self.display_on_canvas(processed_img, "")
            self.info_label.config(text=f"Analysis Complete. Cars: {car_count}, People: {person_count}")

        except Exception as e:
            messagebox.showerror("Processing Error", f"Analysis failed:\n{e}")
            self.info_label.config(text="Error during processing.")
            self.display_on_canvas(self.original_image, "Processing Failed")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficAppGUI(root)
    root.update()
    root.update_idletasks()
    app.display_on_canvas(None, "Load an image to start analysis")
    root.mainloop()