import os
import glob
import torch
import xmltodict
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torch.nn as nn

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotations_dir, transform=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = list(sorted(glob.glob(os.path.join(image_dir, "*.jpg"))))
        self.annotation_files = list(sorted(glob.glob(os.path.join(annotations_dir, "*.xml"))))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        ann_path = self.annotation_files[idx]

        img = Image.open(img_path).convert("RGB")

        with open(ann_path, encoding='utf-8') as f:
            ann = xmltodict.parse(f.read())

        boxes = []
        labels = []
        if 'object' in ann['annotation']:
            objects = ann['annotation']['object']
            if not isinstance(objects, list):
                objects = [objects]

            for obj in objects:
                xmin = int(obj['bndbox']['xmin'])
                xmax = int(obj['bndbox']['xmax'])
                ymin = int(obj['bndbox']['ymin'])
                ymax = int(obj['bndbox']['ymax'])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # Assuming all labels are 1 (container number)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            img = self.transform(img)
        
        return img, target

    def collate_fn(self, batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

# Define transformations
def transform(image):
    return F.to_tensor(image)

# Example usage
dataset = CustomDataset("貨櫃資料集/訓練集", "貨櫃資料集/訓練集_xml", transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)

# Load pretrained SSD model
model = ssdlite320_mobilenet_v3_large(pretrained=True)

# Replace the head of the model to match the number of classes
num_classes = 2  # background and container number
model.head.classification_head.num_classes = num_classes

# Move the model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update learning rate scheduler
    lr_scheduler.step()

    # Print loss (optional)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

# Evaluation loop
model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)
        # Evaluate predictions (calculate accuracy, precision, recall, etc.)
        # Your implementation here

# Save trained model
torch.save(model.state_dict(), 'ssd_model.pth')
