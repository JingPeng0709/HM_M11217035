import os
import glob
import xmltodict
import pytesseract
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

# 載入模型
model = ssdlite320_mobilenet_v3_large(pretrained=True)
num_classes = 2  # 背景和貨櫃號碼
model.head.classification_head.num_classes = num_classes
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load('ssd_model.pth', map_location=device))
model.to(device)
model.eval()

# 測試資料集路徑
test_image_dir = "貨櫃資料集/測試集"
test_annotation_dir = "貨櫃資料集/測試集_xml"

# 讀取測試資料集
test_image_files = list(sorted(glob.glob(os.path.join(test_image_dir, "*.jpg"))))
test_annotation_files = list(sorted(glob.glob(os.path.join(test_annotation_dir, "*.xml"))))

# 計算測試資料集筆數
num_test_samples = len(test_image_files)

# 初始化正確辨識的計數器
correct_recognitions = 0

# 逐個處理測試資料
for img_path, ann_path in zip(test_image_files, test_annotation_files):
    # 讀取圖片
    img = Image.open(img_path).convert("RGB")
    
    # 使用OCR進行文字辨識
    text = pytesseract.image_to_string(img)
    
    # 讀取 XML 標註檔案
    with open(ann_path, encoding='utf-8') as f:
        ann = xmltodict.parse(f.read())
    
    # 獲取貨櫃號碼的第一行文字
    container_number = ann['annotation']['object']['name']
    container_number_first_row = container_number[:11]  # 只取前11個字符
    
    # 比對OCR辨識結果和貨櫃號碼第一行文字是否一致
    if container_number_first_row == text[:11]:
        correct_recognitions += 1

# 計算準確率
accuracy = correct_recognitions / num_test_samples

print(f"測試資料集筆數：{num_test_samples}")
print(f"辨識正確筆數：{correct_recognitions}")
print(f"準確率：{accuracy}")
