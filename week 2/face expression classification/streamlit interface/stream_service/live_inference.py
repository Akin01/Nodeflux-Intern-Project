import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as tt
import time
import streamlit as st
from streamlit_webrtc import VideoTransformerBase
import os

basename = os.path.abspath(os.path.dirname(__file__))
base_model_file = os.path.join(
    basename, "models", "emotion_detection_model_state.pth")

face_classifier = os.path.join(basename, "haarcascade_frontalface_default.xml")

model_state = torch.load(
    f"{base_model_file}", map_location=torch.device('cpu'))

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Suprise"]


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ELU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)

        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(6), nn.Flatten(), nn.Linear(64, num_classes)
        )

    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)

        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)

        return self.classifier(out)


model = ResNet(1, len(class_labels))
model.load_state_dict(model_state)

# load face
try:
    face_cascade = cv2.CascadeClassifier(face_classifier)
except Exception:
    st.write("Error loading cascade classifiers")


class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = tt.functional.to_pil_image(roi_gray)
                roi = tt.functional.to_grayscale(roi)
                roi = tt.ToTensor()(roi).unsqueeze(0)

                #  make a prediction on the ROI
                t0 = time.time()
                tensor = model(roi)
                inference_time = time.time() - t0

                probs = torch.nn.functional.softmax(tensor, dim=1)
                conf, num_class = torch.max(probs, 1)
                label = class_labels[num_class.item()]

                label_position = (x, y)

                cv2.putText(img, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(img, f"confidence score : {round(conf.item(), 2)}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                cv2.putText(img, f"inference time : {round(inference_time, 2)}", (50, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            else:
                cv2.putText(img, "No Face Found", (20, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3, )

        return img
