from module.irislandmarks import IrisLandmarks
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from skimage import measure
import streamlit as st

from module.iris_recog import *
from module.img_test import *
from module.img_process import *
from module.img_rotate import *
from module.densenet_seg.test import *
from module.eye_recog import *
from module.mediapipe_eye import *


def method_1(img_num, img_side, img_take):
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(
        f'C:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/main/Iris-Dataset/CASIA-Iris-Thousand/{str(img_num).zfill(3)}/{img_side}/S5{str(img_num).zfill(3)}{img_side}{str(img_take).zfill(2)}.jpg')
    img, angle = rotate_image(Image.fromarray(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (64, 64))
    net = IrisLandmarks().to(gpu)
    net.load_weights("model/irislandmarks.pth")
    eye_gpu, iris_gpu = net.predict_on_image(img_resized)
    eye = eye_gpu.cpu().numpy()
    iris = iris_gpu.cpu().numpy()

    ratio_h = int(img.shape[0] / 64)
    ratio_w = int(img.shape[1] / 64)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(iris[:, :, 0]*ratio_w, iris[:, :, 1]*ratio_h, zorder=2, s=10)
    ax.scatter(eye[:, :, 0]*ratio_w, eye[:, :, 1]*ratio_h, zorder=2, s=10)
    ax.imshow(img, zorder=1)
    st.pyplot(fig)


def method_2(img_num, img_side, img_take, angle):
    model_name = 'densenet'
    model_path = 'model/densenet_seg.pkl'
    img = read_image(
        f'C:/Users/jimyj/Desktop/TAIST/Thesis/Source_Code/main/Iris-Dataset/CASIA-Iris-Thousand/{str(img_num).zfill(3)}/{img_side}/S5{str(img_num).zfill(3)}{img_side}{str(img_take).zfill(2)}.jpg')
    img_rot, angle = rotate_image(
        Image.fromarray(img), expand=False, angle=angle)
    img_rot, img_seg = run_prediction(
        img_rot, model_name, model_path, use_gpu=True)

    # Assuming img_seg is your image array
    contours_eye = np.array(
        [max(measure.find_contours(img_seg, 0.3), key=len)])
    contours_iris = np.array(
        [max(measure.find_contours(img_seg, 0.6), key=len)])
    contours_pupil = np.array(
        [max(measure.find_contours(img_seg, 0.9), key=len)])

    # Draw the contours on the original image
    fig, ax = plt.subplots()
    ax.imshow(img_rot, cmap='gray')

    for contour in contours_eye:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='green')
    for contour in contours_iris:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    for contour in contours_pupil:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='blue')
    st.pyplot(fig)
