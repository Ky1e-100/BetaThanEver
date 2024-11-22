import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image

# Adjust the path to your YOLOv5 directory
yolov5_path = os.path.join(os.path.dirname(__file__), "../yolov5")
sys.path.append(yolov5_path)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes

device = select_device('0' if torch.cuda.is_available() else 'cpu')

model_path = 'ML/Model/best.pt' 
model = DetectMultiBackend(model_path, device=device)
model.eval()

def letterbox_image(image, desired_size=(640, 640)):
    ih, iw = image.shape[:2]
    w, h = desired_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    new_image = np.full((h, w, 3), 114, dtype=np.uint8)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    new_image[dy:dy + nh, dx:dx + nw, :] = image_resized

    return new_image, scale, dx, dy

def detect_holds(image_path):
    img0 = cv2.imread(image_path)  # Original image
    assert img0 is not None, f"Image not found at {image_path}"

    # Resize image with aspect ratio preserved
    img, scale, dx, dy = letterbox_image(img0, desired_size=(640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0 
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)

    # Remove duplicate detections
    pred = non_max_suppression(pred)[0]

    holds = []
    if pred is not None and len(pred):
        # Adjust boxes from padded image back to original image size
        pred[:, 0] -= dx
        pred[:, 1] -= dy
        pred[:, 2] -= dx
        pred[:, 3] -= dy
        pred[:, :4] /= scale

        # Clip boxes to image dimensions
        pred[:, 0].clamp_(0, img0.shape[1])
        pred[:, 1].clamp_(0, img0.shape[0])
        pred[:, 2].clamp_(0, img0.shape[1])
        pred[:, 3].clamp_(0, img0.shape[0])

        for idx, (*xyxy, conf, cls) in enumerate(pred):
            xyxy = [float(x.item()) for x in xyxy]
            conf = float(conf.item())
            cls = int(cls.item())
            hold = {
                'id': idx,
                'class': model.names[cls],  # Get class name
                'confidence': conf,
                'box': xyxy,  # [xmin, ymin, xmax, ymax]
                'center': [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2],
            }
            holds.append(hold)
    else:
        print("No holds detected.")

    return holds

def filter_holds(holds, target_class):
    filtered_holds = [hold for hold in holds if hold['class'] == target_class]
    return sorted(filtered_holds, key=lambda x: x['center'][1], reverse=True) 

def draw_holds(image_path, holds):
    image = cv2.imread(image_path)
    assert image is not None, f"Image not found at {image_path}"
    output_image = image.copy()
    box_colour = (0, 0, 255)
    box_thickness = 4
    label_colour = (255, 0, 0)
    count = 1
    for idx, hold in enumerate(holds, start=1):
        x1, y1, x2, y2 = map(int, hold['box'])
        cv2.rectangle(output_image, (x1, y1), (x2, y2), box_colour, box_thickness)
        label = str(count)
        cv2.putText(output_image, label, (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 5, label_colour, 2)
        count += 1
    return output_image

def display_image(image, window_name='Image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test
if __name__ == "__main__":
    image_path = 'dataset/test/test2.jpg'
    holds = detect_holds(image_path)

    # Filter holds by class if needed
    target_class = "Pink"
    filtered_holds = filter_holds(holds, target_class)

    image = draw_holds(image_path, filtered_holds)
    if image is not None:
        # display_image(image)
        cv2.imwrite('output_image.jpg', image)
        print("Output image saved as 'output_image.jpg'")
