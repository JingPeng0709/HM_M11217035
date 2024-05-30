import os
import glob
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import cv2
import numpy as np

# Load the SSD model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
num_classes = 2  # background and container number
model.head.classification_head.num_classes = num_classes

# Load the trained weights
model.load_state_dict(torch.load('ssd_model.pth'))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the transformation
def transform(image):
    return F.to_tensor(image)

# Function to preprocess the frame
def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)
    return img

# Function to draw bounding boxes
def draw_boxes(frame, boxes, labels, scores, threshold=0.5):
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to process video
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess(frame)
        img = img.to(device)

        with torch.no_grad():
            prediction = model(img)[0]

        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        draw_boxes(frame, boxes, labels, scores)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process the videos
video_folder = '影片資料集'
output_folder = 'processed_videos'
os.makedirs(output_folder, exist_ok=True)

video_paths = glob.glob(f'{video_folder}/*.avi')
for video_path in video_paths:
    output_path = os.path.join(output_folder, os.path.basename(video_path))
    process_video(video_path, output_path)
    print(f'Processed {video_path} and saved to {output_path}')
