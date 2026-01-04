# Cell 1: Install Libraries and Mount Google Drive
import os
import shutil
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xml.etree.ElementTree as ET
from IPython.display import display, Javascript
from google.colab import drive # Import Google Drive library

print("--- Installing required libraries ---")
!pip install ultralytics
!pip install opencv-python matplotlib lxml # lxml for XML processing (for Pascal VOC)
print("Required libraries installed successfully.")

# Mount Google Drive
print("\n--- Mounting Google Drive ---")
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# Define your project directory on Google Drive
# All training results (weights, logs, plots) will be saved here.
# If this directory exists and contains previous checkpoints, YOLOv8 will resume training.
GOOGLE_DRIVE_PROJECT_PATH = '/content/drive/MyDrive/YOLOv8_Car_Detection_Project'
# A specific run name to organize results within the project path
YOLO_RUN_NAME = 'train_cars_resume_run'

print(f"Training results will be saved to: {GOOGLE_DRIVE_PROJECT_PATH}/{YOLO_RUN_NAME}")


# Cell 2: Download and Prepare Pascal VOC Dataset (Filtering Classes)
DATASET_ROOT_DIR = 'datasets_pascal_voc_cars_only_quick' # Changed folder name for clarity
print(f"\n--- Downloading and preparing Pascal VOC dataset for cars only (QUICK VERSION) ---")
!mkdir -p {DATASET_ROOT_DIR}
%cd {DATASET_ROOT_DIR}

# --- Download Pascal VOC 2007 (train/val only) ---
print("Downloading Pascal VOC 2007 train/val...")
!wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -O VOCtrainval_06-Nov-2007.tar
!tar -xf VOCtrainval_06-Nov-2007.tar
print("Pascal VOC 2007 train/val downloaded and extracted.")

VOC2007_PATH = os.path.join(os.getcwd(), 'VOCdevkit/VOC2007')

# --- Prepare folders for YOLO format ---
YOLO_DATA_PATH = os.path.join(os.getcwd(), 'yolo_data')
!mkdir -p {YOLO_DATA_PATH}/images/train
!mkdir -p {YOLO_DATA_PATH}/labels/train
!mkdir -p {YOLO_DATA_PATH}/images/val
!mkdir -p {YOLO_DATA_PATH}/labels/val

# --- Define target classes and convert Pascal VOC to YOLO format ---
# **Our target class: 'car' only**
SELECTED_CLASSES = ['car'] # Includes only 'car'

# Class ID in your fine-tuned model: 'car' = 0
CLASS_NAME_TO_YOLO_ID = {name: i for i, name in enumerate(SELECTED_CLASSES)}
TARGET_CLASS_NAMES_PERSIAN = {
    'car': 'ماشین', # Keeping Persian for display consistency if needed, but primary use is English
}

print(f"\n--- Converting Pascal VOC to YOLO format for selected class: {SELECTED_CLASSES} ---")

def convert_pascal_to_yolo(voc_path, image_set_file, output_images_dir, output_labels_dir, selected_classes_map):
    annotations_path = os.path.join(voc_path, 'Annotations')
    jpeg_images_path = os.path.join(voc_path, 'JPEGImages')
    image_set_path = os.path.join(voc_path, 'ImageSets/Main', image_set_file)

    print(f"Processing image set: {image_set_file} from {voc_path}")

    if not os.path.exists(image_set_path):
        print(f"Warning: {image_set_file} not found at {image_set_path}. Skipping.")
        return

    with open(image_set_path, 'r') as f:
        image_names = [line.strip().split(' ')[0] for line in f if line.strip()]

    num_converted_images = 0
    num_converted_objects = 0

    for img_name in image_names:
        annotation_file = os.path.join(annotations_path, f"{img_name}.xml")
        image_file = os.path.join(jpeg_images_path, f"{img_name}.jpg")

        if not os.path.exists(annotation_file) or not os.path.exists(image_file):
            continue

        try:
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

            yolo_labels = []
            has_selected_object = False
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in selected_classes_map: # This will now only match 'car'
                    has_selected_object = True
                    class_id = selected_classes_map[class_name] # 'car' will be 0

                    bbox = obj.find('bndbox')
                    x_min = int(bbox.find('xmin').text)
                    y_min = int(bbox.find('ymin').text)
                    x_max = int(bbox.find('xmax').text)
                    y_max = int(bbox.find('ymax').text)

                    center_x = (x_min + x_max) / 2 / img_width
                    center_y = (y_min + y_max) / 2 / img_height
                    bbox_width = (x_max - x_min) / img_width
                    bbox_height = (y_max - y_min) / img_height

                    yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
                    num_converted_objects += 1

            if has_selected_object:
                shutil.copy(image_file, os.path.join(output_images_dir, f"{img_name}.jpg"))
                with open(os.path.join(output_labels_dir, f"{img_name}.txt"), 'w') as f:
                    f.write("\n".join(yolo_labels))
                num_converted_images += 1

        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
            continue
    print(f"Converted {num_converted_images} images with {num_converted_objects} objects for selected classes.")

# Execute conversion for VOC 2007 (trainval) - as training set
convert_pascal_to_yolo(VOC2007_PATH, 'trainval.txt',
                       os.path.join(YOLO_DATA_PATH, 'images', 'train'),
                       os.path.join(YOLO_DATA_PATH, 'labels', 'train'),
                       CLASS_NAME_TO_YOLO_ID)

# Execute conversion for VOC 2007 (val) - as validation set
convert_pascal_to_yolo(VOC2007_PATH, 'val.txt',
                       os.path.join(YOLO_DATA_PATH, 'images', 'val'),
                       os.path.join(YOLO_DATA_PATH, 'labels', 'val'),
                       CLASS_NAME_TO_YOLO_ID)

print("\n--- Dataset preparation complete. ---")

# --- Final data.yaml settings for YOLOv8 ---
final_data_yaml_content = f"""
path: {YOLO_DATA_PATH}
train: images/train
val: images/val
nc: {len(SELECTED_CLASSES)}
names: {list(CLASS_NAME_TO_YOLO_ID.keys())}
"""

with open('data.yaml', 'w') as f:
    f.write(final_data_yaml_content)

print(f"data.yaml created successfully for classes: {list(CLASS_NAME_TO_YOLO_ID.keys())}.")
%cd .. # Return to the main Colab folder (/content/)


# Cell 3: Train the YOLOv8 model

# We fine-tune a model that learns only 'car'.
model_fine_tuned = YOLO('yolov8n.pt')

print("\n--- Starting model training ---")
print("Training exclusively for 'car' class with increased epochs for better accuracy.")
print("Training on a more geometrically consistent object like 'car' from Pascal VOC is expected to yield higher accuracy.")
print("Increased epochs (300) and using 640x640 image size will help detect smaller objects like distant cars.")

# Train the model, saving results to Google Drive for resume capability
# 'project' specifies the main directory in Google Drive.
# 'name' specifies a subfolder within 'project' for this specific training run.
# If 'project/name' exists, YOLOv8 will automatically attempt to resume from the last checkpoint.
results = model_fine_tuned.train(data='data.yaml', epochs=500, imgsz=640, device=0, cache=True,
                                 project=GOOGLE_DRIVE_PROJECT_PATH, name=YOLO_RUN_NAME)
print("Model training finished.")

# The trained model (best.pt) will be saved within the specified project/name directory.
# Construct the path to the best model's weights.
# The `model_fine_tuned.trainer.save_dir` will now point to the Google Drive path.
trained_model_path = os.path.join(model_fine_tuned.trainer.save_dir, 'weights', 'best.pt')
print(f"Trained model saved at: {trained_model_path}")


# Cell 4: Function for detection and counting in an image

def count_cars_in_image(image_path, model_path, confidence_threshold=0.5): # Adjusted confidence threshold for potentially smaller objects
    """
    This function counts only cars in a given image and displays the image with detection boxes.
    If no cars are found in the image, it returns 0.
    """
    model = YOLO(model_path) # This is your fine-tuned model that recognizes only cars.

    # Perform prediction on the image for the single class your fine-tuned model knows ('car').
    # A lower confidence_threshold (e.g., 0.3-0.5) can help detect smaller/more distant cars,
    # but might also increase false positives. Experiment with this value!
    predictions = model(image_path, conf=confidence_threshold, iou=0.7, verbose=False)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return 0 # Return 0 cars if image fails to load
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    car_count = 0

    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)

    # Define specific color for cars
    car_color_hex = '#008000' # Green for car

    # Get class names from the fine-tuned model (it will map 0 to 'car')
    class_names_from_model = model.names
    # Get the ID for 'car' from the fine-tuned model (should be 0)
    fine_tuned_car_id = CLASS_NAME_TO_YOLO_ID.get('car', -1)

    for r in predictions:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            # Ensure the detected class is 'car' and confidence is above threshold
            if class_id == fine_tuned_car_id and conf >= confidence_threshold:
                detected_class_english_name = class_names_from_model.get(class_id)
                if detected_class_english_name == 'car': # Double check
                    car_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    width = x2 - x1
                    height = y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height,
                                             linewidth=2, edgecolor=car_color_hex, facecolor='none')
                    ax.add_patch(rect)

                    # Display name and confidence
                    display_name = detected_class_english_name # Using English name
                    ax.text(x1, y1 - 10, f"{display_name}: {conf:.2f}",
                            color='white', fontsize=10, bbox=dict(facecolor=car_color_hex, alpha=0.7))

    # Generate title based on car count
    title_text = ""
    if car_count > 0:
        title_text = f"Detected Cars: {car_count}"
    else:
        title_text = "No cars detected in the image."


    ax.set_title(title_text)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return car_count

# Cell 5: Main flow for upload and prediction

def upload_files_to_colab():
    from google.colab import files
    uploaded = files.upload()
    if uploaded:
        filename = list(uploaded.keys())[0]
        print(f"File '{filename}' uploaded successfully.")
        return filename
    else:
        print("No file uploaded.")
        return None

def main_counting_flow():
    print("\n--- Ready to upload image and count cars ---")
    print("Please upload an image where you want to count cars.")

    uploaded_image_path = upload_files_to_colab()

    if uploaded_image_path:
        print(f"Processing uploaded image: '{uploaded_image_path}'")
        # Set a confidence_threshold. For detecting distant/small objects, you might need
        # to lower this slightly (e.g., 0.3 or 0.4), but be aware of potential false positives.
        # A value of 0.5 is a good starting point for general car detection.
        car_count_result = count_cars_in_image(uploaded_image_path, trained_model_path, confidence_threshold=0.5)
        print("\nDetected car count:")
        print(f"- Cars: {car_count_result}")
    else:
        print("No file uploaded. Please upload an image to perform counting.")

# Cell 6: Execute the entire process in Colab

if __name__ == "__main__":
    # **IMPORTANT:** Before running the code, make sure to enable GPU in Colab:
    # Runtime -> Change runtime type -> Hardware accelerator -> GPU

    main_counting_flow()
