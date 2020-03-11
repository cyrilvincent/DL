from imageai.Detection import ObjectDetection
import os

"""
Modify import __init__.py
import cv2
from imageai.Detection.keras_retinanet.models.resnet import resnet50_retinanet
from imageai.Detection.keras_retinanet.utils.image import read_image_bgr, read_image_array, read_image_stream, \
    preprocess_image, resize_image
from imageai.Detection.keras_retinanet.utils.visualization import draw_box, draw_caption
from imageai.Detection.keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import tensorflow as tf
import os
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.layers import Input
from PIL import Image
import colorsys

Modify import models.py
from functools import wraps
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPool2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.compat.v1.keras.layers import LeakyReLU, Input
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.regularizers import l2
from tensorflow.compat.v1.keras.models import Model
"""
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="data/img/familly.jpg", output_image_path="data/img/famillynew.jpg", minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")