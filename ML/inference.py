import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image

yolov5_path = os.path.join(os.path.dirname(__file__), '../yolov5')
sys.path.append(yolov5_path)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes

device = select_device('0' if torch.cuda.is_available() else 'cpu')

model_path = 'ML/Model/best.pt' 
model = DetectMultiBackend(model_path, device=device)
model.eval()

def detect_holds(image_path):
    img0 = cv2.imread(image_path)
    assert img0 is not None, f"Image not found at {image_path}"

    # Prep image
    img = cv2.resize(img0, (640, 640))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize to [0, 1]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)

    # Apply Non-Maximum Suppression (NMS) -- Removing duplicate detections
    pred = non_max_suppression(pred)[0]

    holds = []
    if pred is not None and len(pred):
        # Rescale boxes to original image size
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()

        for idx, (*xyxy, conf, cls) in enumerate(pred):
            xyxy = [x.item() for x in xyxy]
            conf = conf.item()
            cls = int(cls.item())
            hold = {
                'id': idx,
                'class': model.names[cls],  # Get class name
                'confidence': conf,
                'box': xyxy,  # [xmin, ymin, xmax, ymax]
                'center': [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
            }
            holds.append(hold)
    else:
        print("No holds detected.")

    return holds

# test
if __name__ == "__main__":
    image_path = 'dataset//test/test1.jpg'
    holds = detect_holds(image_path)
    for hold in holds:
        print(f"Hold ID: {hold['id']}, Class: {hold['class']}, Confidence: {hold['confidence']:.2f}")
        print(f"Bounding Box: {hold['box']}")
        print(f"Center Coordinates: {hold['center']}")
        print("-----------")
