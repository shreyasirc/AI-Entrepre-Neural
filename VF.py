import os
import glob
import time
from PIL import Image
from os import listdir
from shutil import copyfile
from os.path import isfile, join
from matplotlib import pyplot as plt
import cv2

Height = 224
Width  = 224
BatchSize = 24
lr_rate=.0015
Version = 5
load_model = True
accuracy = 0
accuracyCount = 0
trainableCount = 30

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tflite_runtime.interpreter as tflite

def predict_vf(imagepath):
    
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    )       
    # detect only the facial region
    crop = image.copy()                    
    for (x, y, w, h) in faces:                  
        crop = image[y:y+h, x:x+w]
        cv2_imshow(crop)

    # crop the detected face
    crop = cv2.resize(crop, (224, 224))             
    image = crop

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]),)      # reshape image

    
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="models/model.tflite")
    interpreter.allocate_tensors()

    image = image.astype('float32')

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    Y_pred = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    return Y_pred[0]                                        # return autistic probability