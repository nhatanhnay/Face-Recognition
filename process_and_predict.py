import os
import cv2
import torch
import numpy as np
from device import device
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

classes = os.listdir(
    r'C:\Users\lecon\OneDrive\Máy tính\Face Regconition\archive\Celebrity Faces Dataset')


def process_and_predict(image_path, model):
    # Load the image
    img = Image.open(image_path)

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Detect faces using the face_cascade
    faces = face_cascade.detectMultiScale(img_cv, 1.1, minNeighbors=15)

    for i, (x, y, w, h) in enumerate(faces):

        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_img = img_cv[y:y + h, x:x + w]

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_data = transform(Image.fromarray(face_img)
                               ).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_data)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

        predicted_class = preds.item()
        probability = probs[0][predicted_class].item() * 100
        font_scale = min(w, h) / 200

        label = f'{classes[preds.item()]}: {probability:.2f}%'
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(img_cv, label, (x, y - 10 -
                    label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.show()
