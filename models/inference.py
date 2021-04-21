from PIL import Image
import numpy as np
import sys
import pathlib
import glob
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import keras
from sklearn.metrics import classification_report

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IEPlugin
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

def pre_process_image(imagePath, img_height=224):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    image = Image.open(imagePath)
    #processedImg = image.resize((h, w), resample=Image.ANTIALIAS)

    # Normalize to keep data between 0 - 1
    #processedImg = (np.array(processedImg) - 0) 
    processedImg = keras.preprocessing.image.img_to_array(image)
    # Change data layout from HWC to CHW
    processedImg = processedImg.transpose((2, 0, 1))
    processedImg = processedImg.reshape((n, c, h, w))

    return processedImg



# Plugin initialization for specified device and load extensions library if specified.
plugin_dir = None
model_xml = '/content/drive/MyDrive/Kaggle-Autism/models/h5/timestamp/frozen_graph.xml'
model_bin = '/content/drive/MyDrive/Kaggle-Autism/models/h5/timestamp/frozen_graph.bin'
# Devices: GPU (intel), CPU, MYRIAD
plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
# Read IR
net = IENetwork(model=model_xml, weights=model_bin)
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 1
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
# Load network to the plugin
exec_net = plugin.load(network=net)
del net

y_true = []
#Prepare Test Data
test_data_gen=[]
file_list=glob.glob("/content/drive/MyDrive/Kaggle-Autism/test/autistic/*.jpg")
for i in file_list:
  processedImg = pre_process_image(i)
  test_data_gen.append(processedImg)
  y_true.append('autistic')
file_list=glob.glob("/content/drive/MyDrive/Kaggle-Autism/test/non_autistic/*.jpg")
for i in file_list:
  processedImg = pre_process_image(i)
  test_data_gen.append(processedImg)
  y_true.append('non_autistic')
  




# Shuffle Data
#random.shuffle(test_data_gen)
print("len(test_data_gen)):")
print(len(test_data_gen))

y_pred = []
# Run inference
start_time = time.time()
for i in range(len(test_data_gen)):
    predictions = exec_net.infer(inputs={input_blob: test_data_gen[i]})

    print(predictions)
    
  
    if(predictions['model/classifier/Softmax'][0][0]<0.5):
      print("Non-autistic     "+str(abs(predictions['model/classifier/Softmax'][0][1]*100))+"%")
      y_pred.append('non_autistic')
      
    else:  
      print("Autistic    "+str(100-abs(predictions['model/classifier/Softmax'][0][1]*100))+"%")
      y_pred.append('autistic')


    
    
print("Total Time: ", time.time()-start_time)
print("Average Time Per Image: ", (time.time()-start_time)/len(test_data_gen))
target_names = ['autistic', 'non_autistic']
print("Classification report: ", classification_report(y_true, y_pred, target_names=target_names))
