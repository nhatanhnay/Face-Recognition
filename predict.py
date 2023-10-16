from process_and_predict import process_and_predict
from load_model import load_matched_state_dict
import torch
from device import device
import os
from facenet_pytorch import InceptionResnetV1
classes_path = r'C:\Users\lecon\OneDrive\Máy tính\Face Regconition\archive\Celebrity Faces Dataset'
classes = os.listdir(classes_path)
model_path = r'C:\Users\lecon\OneDrive\Máy tính\Face Regconition\Face_Recognition_checkpoint.pth'
resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(classes)
).to(device)

load_matched_state_dict(resnet, torch.load(model_path))
resnet.eval()

image_path = r'C:\Users\lecon\OneDrive\Máy tính\Face Regconition\176574779_510338043461216_8700602942414461949_n.jpg'
process_and_predict(image_path, resnet)
