import tensorflow as tf
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import cv2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from pathlib import Path
np.set_printoptions(threshold=np.inf)



# model=tf.keras.models.load_model('VGG16.h5')
GTSRB = tf.keras.datasets.GTSRB
(train_x,train_y),(test_x,test_y) = GTSRB.load_data()
