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

path = "/content/drive/MyDrive/Casillo&Natale/dataset_Spagnolo_giapponese/train"    #Path della cartella contenente i video da 15 secondi da computare
destinationPath = "/content/drive/MyDrive/Casillo&Natale/roba gaetano/test"  #Path della cartella di destinazione per i video csv
video_destination_path = "/content/drive/MyDrive/Casillo&Natale/dataset_4_7/butta"  #path della cartella di destinazione per i video con presenti solo le labbra
video_destination_path_land = "/content/drive/MyDrive/Casillo&Natale/dataset_4_7/butta"


os.chdir(destinationPath)  


#for videoFile in tqdm(os.listdir(path)):     #per ogni file video nella cartella
  


for videoFile in tqdm(os.listdir(path)):     #per ogni file video nella cartella
    print("-----------Inizio computazione " + videoFile + "----------------")
    frame = cv2.imread(path + "/" + videoFile)
    print(frame)
    print(videoFile)
    ds_factor = 0.5
    
    while(1):
          blur = cv2.GaussianBlur(frame, (5, 5), 0)
          
          gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
          ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
          #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
          x=0
          y=0
          w=0
          h=0
          cv2.imwrite(videoFile+"gaus.jpg", thresh )
          # Further noise removal
          kernel = np.ones((3, 3), np.uint8)
          opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

          # sure background area
          sure_bg = cv2.dilate(opening, kernel, iterations=3)

          # Finding sure foreground area
          dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
          ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

          # Finding unknown region
          sure_fg = np.uint8(sure_fg)
          unknown = cv2.subtract(sure_bg, sure_fg)
          cv2.imwrite(videoFile+"sfggauss.jpg", sure_fg )

          ret, markers = cv2.connectedComponents(sure_fg)

          # Add one to all labels so that sure background is not 0, but 1
          markers = markers + 1

          # Now, mark the region of unknown with zero
          markers[unknown == 255] = 0

          markers = cv2.watershed(blur, markers)
          blur[markers == -1] = [255, 0, 0]

          cv2.imwrite(videoFile+"markersgauss.jpg", markers )
          cv2.imwrite(videoFile+"framegauss.jpg", blur )


         
          y = int(y - 0.15*h)
          #cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0),0)

          #track_window = (x, y, w, h)
          #roi = frame
          #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
          #ret2, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
          #res = cv2.bitwise_and(roi, roi, mask=mask)
         # hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
         # roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 255], [0, 180, 0, 255])
         # cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
         # term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
         # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
         # dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 255], 1)
            # apply meanshift to get the new location
          #ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            # Draw it on image
          #x, y, w, h = track_window
          #img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 0, 1)
          #lip = frame[y:y + h, x:x + w]
          #video = cv2.resize(lip, (120,120), interpolation=cv2.INTER_AREA)
          #cv2.imwrite(videoFile, res )
          break
    break
        