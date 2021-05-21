import cv2
import os
import numpy as np
import glob
import dlib
import csv
from google.colab.patches import cv2_imshow
import math
import time
import shutil
import psutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from tqdm import tqdm
from collections import OrderedDict

def shape_to_np(shape, dtype="int"):
    """
    Restituisce le una lista contente le coppie
    di coordinate che rappresentano i landmark
    """
    
    coords = np.zeros((68, 2), dtype=dtype)  #inizializza la lista
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
# Definisce un dizionario che mappa gli indici per i
# landmark facciali per ogni regione del viso
FACIAL_LANDMARKS_IDXS = OrderedDict([
	  ("mouth", (48, 68)),
    ("mouth_intern", (60, 68)),
    ("mouth_extern", (48, 60)),
	  ("right_eyebrow", (17, 22)),
	  ("left_eyebrow", (22, 27)),
	  ("right_eye", (36, 42)),
	  ("left_eye", (42, 48)),
	  ("nose", (27, 35)),
	  ("jaw", (0, 17))
])

SIZE = (300, 200)

detector = dlib.get_frontal_face_detector()  #inizializza il face detector(HOG-based) della libreria dlib
predictor = dlib.shape_predictor("/content/drive/MyDrive/shape_predictor_68_face_landmarks.dat") 

path = "/content/drive/MyDrive/Casillo&Natale/roba gaetano/train"    #Path della cartella contenente i video da 15 secondi da computare
destinationPath = "/content/drive/MyDrive/Casillo&Natale/roba gaetano/test"  #Path della cartella di destinazione per i video csv
video_destination_path = "/content/drive/MyDrive/Casillo&Natale/dataset_4_7/butta"  #path della cartella di destinazione per i video con presenti solo le labbra
video_destination_path_land = "/content/drive/MyDrive/Casillo&Natale/dataset_4_7/butta"


os.chdir(destinationPath)  


for videoFile in tqdm(os.listdir(path)):     #per ogni file video nella cartella
  


  for videoFile in os.listdir(path):     #per ogni file video nella cartella
    print("-----------Inizio computazione " + videoFile + "----------------")
    cap = cv2.VideoCapture(path + "/" + videoFile)
    ds_factor = 0.5
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        i=i+1
        if ret == False:
            break
        rects = detector(frame, 1)
        for rect in rects: 
          shape = predictor(frame, rect)    #Determina i landmark del viso
          shape = shape_to_np(shape)   
          #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        

          (x, y, w, h) = cv2.boundingRect(np.array([shape[FACIAL_LANDMARKS_IDXS["mouth"][0]:FACIAL_LANDMARKS_IDXS["mouth"][1]]]))
          y = int(y - 0.15*h)
          cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

          track_window = (x, y, w, h)
          roi = frame[y:y + h, x:x + w]
          gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
          ret2, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
          res = cv2.bitwise_and(roi, roi, mask=mask)
          hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
          roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 255], [0, 180, 0, 255])
          cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
          term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
          hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 255], 1)
            # apply meanshift to get the new location
          ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            # Draw it on image
          x, y, w, h = track_window
          img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 0, 1)
          lip = frame[y:y + h, x:x + w]
          video = cv2.resize(lip, (64,64), interpolation=cv2.INTER_CUBIC)
          cv2.imwrite(videoFile + "_frame%d.jpg" %i , roi )
          break
        break

          

  

        

    #print("-----------Conclusa computazione " + videoFile + "----------------")
            #video = cv2.resize(lip, (64, 64), interpolation=cv2.INTER_CUBIC)
            #file_tosave ='/content/drive/MyDrive/Casillo&Natale/roba gaetano/test' + videoFile+"_frame.jpg"
            #cv2.imwrite(file_tosave, video)
            


def shape_to_np(shape, dtype="int"):
    """
    Restituisce le una lista contente le coppie
    di coordinate che rappresentano i landmark
    """
    
    coords = np.zeros((68, 2), dtype=dtype)  #inizializza la lista
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
