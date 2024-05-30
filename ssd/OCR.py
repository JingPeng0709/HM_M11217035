import cv2
import pytesseract
import os
import glob

# 指定 Tesseract 執行檔的位置
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Tang\anaconda3\envs\MLtest\tesseract.exe'

# OCR 設定
custom_config = r'--oem 3 --psm 6'

# Function to perform OCR on a single frame
def ocr_frame(frame):
    # 將幀轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 進行 OCR
    text = pytesseract.image_to_string(gray, config=custom_config)
    return text

# Function to process video and extract text using OCR
def process_video_ocr(video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    all_text = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform OCR on the frame
        text = ocr_frame(frame)
        all_text.append(text)

    cap.release()

    # Save the extracted text to a file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_text))

    print(f'Saved OCR results to {output_txt_path}')

# Process all videos in the directory
video_folder = 'processed_videos'
output_folder = 'ocr_results'
os.makedirs(output_folder, exist_ok=True)

video_paths = glob.glob(f'{video_folder}/*.avi')
for video_path in video_paths:
    output_txt_path = os.path.join(output_folder, os.path.basename(video_path).replace('.avi', '.txt'))
    process_video_ocr(video_path, output_txt_path)
    print(f'Processed {video_path} and saved OCR results to {output_txt_path}')
