import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from glob import glob
import xml.etree.ElementTree as xet
from sklearn.model_selection import train_test_split
import shutil
from ultralytics import YOLO
import easyocr
import torch

def parse_xml_to_dataframe(dataset_path):
    """Parse Kaggle XML annotations to Pandas DataFrame for YOLO training."""
    labels_dict = {
        'img_path': [], 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': [],
        'img_w': [], 'img_h': []
    }
    xml_files = glob(f'{dataset_path}/annotations/*.xml')

    for filename in sorted(xml_files, key=lambda x: int(re.search(r'\d+', x).group(0) or 0)):
        info = xet.parse(filename)
        root = info.getroot()
        member_object = root.find('object')
        labels_info = member_object.find('bndbox')
        img_name = root.find('filename').text
        img_path = os.path.join(dataset_path, 'images', img_name)

        labels_dict['img_path'].append(img_path)
        labels_dict['xmin'].append(int(labels_info.find('xmin').text))
        labels_dict['xmax'].append(int(labels_info.find('xmax').text))
        labels_dict['ymin'].append(int(labels_info.find('ymin').text))
        labels_dict['ymax'].append(int(labels_info.find('ymax').text))

        height, width, _ = cv2.imread(img_path).shape
        labels_dict['img_w'].append(width)
        labels_dict['img_h'].append(height)

    return pd.DataFrame(labels_dict)

def create_yolo_folders(dataframe, split_name, output_dir):
    """Convert DataFrame to YOLO format (images + labels) for training."""
    labels_path = os.path.join(output_dir, 'cars_license_plate_new', split_name, 'labels')
    images_path = os.path.join(output_dir, 'cars_license_plate_new', split_name, 'images')
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)

    for _, row in dataframe.iterrows():
        img_name, img_extension = os.path.splitext(os.path.basename(row['img_path']))
        x_center = (row['xmin'] + row['xmax']) / 2 / row['img_w']
        y_center = (row['ymin'] + row['ymax']) / 2 / row['img_h']
        width = (row['xmax'] - row['xmin']) / row['img_w']
        height = (row['ymax'] - row['ymin']) / row['img_h']

        label_path = os.path.join(labels_path, f'{img_name}.txt')
        with open(label_path, 'w') as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        shutil.copy(row['img_path'], os.path.join(images_path, f'{img_name}{img_extension}'))

    print(f"Created {split_name} folders: {images_path}, {labels_path}")

def train_yolo_model(data_yaml, epochs=5, img_size=320, batch_size=16):
    """Train YOLOv8 model on custom dataset."""
    model = YOLO('yolov8s.pt')
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache=True
    )
    return model

class LicensePlateRecognizer:
    """Class to detect and recognize license plates using YOLOv8 and EasyOCR."""
    def __init__(self, model, ocr_reader):
        self.model = model
        self.reader = ocr_reader
        self.cropped_images = []

    def extract_plates(self, image_path):
        """Detect license plates and crop them from the input image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(image_path, device='cpu')
        self.cropped_images = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_image = result.orig_img[y1:y2, x1:x2]
                self.cropped_images.append(cropped_image)
        return self.cropped_images

    def recognize_text(self):
        """Extract text from cropped license plate images."""
        if not self.cropped_images:
            return ["No plates detected"]
        text = self.reader.readtext(self.cropped_images[0], detail=0)
        return text if text else ["No text detected"]

    def visualize(self, image_path, output_path=None):
        """Visualize detection results with bounding boxes and save/display."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(image_path, device='cpu')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        plt.imshow(image)
        plt.axis('off')
        plt.title(f'License Plate Detection: {os.path.basename(image_path)}')
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()

        # Visualize cropped plate if available
        if self.cropped_images:
            plt.figure()
            plt.imshow(self.cropped_images[0])
            plt.axis('off')
            plt.title('Cropped License Plate')
            if output_path:
                cropped_path = os.path.join(os.path.dirname(output_path), f'cropped_{os.path.basename(image_path)}')
                cv2.imwrite(cropped_path, self.cropped_images[0])
                print(f"Saved cropped plate to {cropped_path}")
            else:
                plt.show()

def main(args):
    """Main function to run license plate recognition."""
    # Initialize OCR reader
    reader = easyocr.Reader(['en'])

    if args.train:
        # Prepare dataset
        dataset_path = args.dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Parse XML to DataFrame
        alldata = parse_xml_to_dataframe(dataset_path)

        # Split dataset
        train, test = train_test_split(alldata, test_size=0.1, random_state=42)
        train, val = train_test_split(train, train_size=8/9, random_state=42)

        # Create YOLO format folders
        output_dir = 'datasets'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        create_yolo_folders(train, 'train', output_dir)
        create_yolo_folders(val, 'val', output_dir)
        create_yolo_folders(test, 'test', output_dir)

        # Write YOLO dataset YAML
        datasets_yaml = f'''
        path: cars_license_plate_new
        train: train/images
        val: val/images
        test: test/images
        nc: 1
        names: ['license_plate']
        '''
        with open('datasets.yaml', 'w') as f:
            f.write(datasets_yaml)

        # Train YOLO model
        model = train_yolo_model('datasets.yaml', epochs=args.epochs)
    else:
        # Load pre-trained model
        model = YOLO(args.model_path)

    # Initialize recognizer
    recognizer = LicensePlateRecognizer(model, reader)

    # Process multiple images if provided
    image_paths = args.image_paths if args.image_paths else [args.image_path]
    for idx, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Skipping {image_path}: File not found")
            continue

        # Extract and recognize plates
        cropped_images = recognizer.extract_plates(image_path)
        text = recognizer.recognize_text()
        print(f"Image: {image_path}")
        print(f"Detected License Plate Text: {text}")

        # Visualize results
        output_path = args.output_path
        if output_path and len(image_paths) > 1:
            output_path = f"{os.path.splitext(args.output_path)[0]}_{idx}{os.path.splitext(args.output_path)[1]}"
        recognizer.visualize(image_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="License Plate Recognition with YOLOv8 and EasyOCR")
    parser.add_argument('--image-path', type=str, default='test.jpg', help='Path to single input image')
    parser.add_argument('--image-paths', type=str, nargs='+', help='Paths to multiple input images')
    parser.add_argument('--model-path', type=str, default='yolov8s.pt', help='Path to YOLO model weights')
    parser.add_argument('--dataset-path', type=str, default='car-plate-detection', help='Path to Kaggle dataset')
    parser.add_argument('--train', action='store_true', help='Train YOLO model on dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--output-path', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()

    main(args)
