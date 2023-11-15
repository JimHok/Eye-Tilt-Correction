import numpy as np
from matplotlib import pyplot as plt
from module.recog import *
from PIL import Image
import random
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import concurrent.futures
from tqdm.auto import tqdm
import os


def rotate_image(img, angle=None, expand=True):
    # Open the image file
    image = img

    reference_color = image.getpixel((image.size[0] / 2, 0))

    # Rotate the image by a random degree
    if angle == None:
        angle = random.randint(-90, 90)
    rotated_image = image.rotate(angle, expand=expand)

    return np.array(rotated_image), angle


def crop_image(ref_img, img):
    height_ref, width_ref = np.array(ref_img).shape
    height_img, width_img = np.array(img).shape

    left = (width_img - width_ref) / 2
    top = (height_img - height_ref) / 2
    right = (width_img + width_ref) / 2
    bottom = (height_img + height_ref) / 2

    cropped_image = img.crop((left, top, right, bottom))

    return cropped_image
