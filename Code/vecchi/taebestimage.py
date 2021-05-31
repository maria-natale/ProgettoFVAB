# import the necessary packages
import argparse
import time
import cv2
import tqdm
import os

#So, you can load any .py file from a path and set the module name to be whatever you want. So just adjust the module_name to be whatever custom name you'd like the module to have upon importing.

#To load a package instead of a single file, file_path should be the path to the package's root __init__.py

path = "/content/drive/MyDrive/Casillo&Natale/dataset_Spagnolo_giapponese/train"    #Path della cartella contenente i video da 15 secondi da computare
destinationPath = "/content/drive/MyDrive/Casillo&Natale/roba gaetano/test"
os.chdir(destinationPath)  

# construct the argument parser and parse the arguments
model="ESPCN_x4.pb"
for videoFile in os.listdir(path):
  image=cv2.imread(path + "/" + videoFile)


  modelName = model.split(os.path.sep)[-1].split("_")[0].lower()
  modelScale = model.split("_x")[-1]
  modelScale = int(modelScale[:modelScale.find(".")])
  sr = cv2.dnn_superres.DnnSuperResImpl_create()
  cv2.
  sr.readModel(model)
  sr.setModel(modelName, modelScale)
  # load the input image from disk and display its spatial dimensions
  upscaled = sr.upsample(image)

# show the spatial dimensions of the super resolution image
  cv2.imwrite(videoFile+"frameup.jpg", upscaled )
  break
