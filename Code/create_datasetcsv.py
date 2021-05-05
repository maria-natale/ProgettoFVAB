import glob
import cv2 
import os
import numpy as np
import glob
import dlib
from collections import OrderedDict
import csv
from tqdm import tqdm
import pandas as pd

path_git = '/content/ProgettoFVAB/file_dataset'
filename = 'spagnolo_giapponese'
path_drive = '/content/drive/MyDrive/Casillo&Natale'
dataset_dir = 'dataset_Spagnolo_giapponese'

detector = dlib.get_frontal_face_detector()  #inizializza il face detector(HOG-based) della libreria dlib
predictor = dlib.shape_predictor("/content/drive/MyDrive/Predictor68Landmarks/shape_predictor_68_face_landmarks.dat")  #crea il predictor per i landmark del viso

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
SIZE = (300, 200)  #Dimensione a cui verranno stampati i video della labbra 

# Prende una stringa nome ed una matrice di distanze euclidee
# e stampa la matrice 
def print_csv_file(name, matrix, file_to_create):
    """
    Questa funzione prende una stringa nome ed una matrice
    di distanze euclidee e stampa su file .csv la matrice
    """
    os.chdir(os.path.join(path_drive, dataset_dir, file_to_create))
    with open(str(name)+".csv", mode='w', newline='') as csv_file:  #apre il file csv
        writer = csv.writer(csv_file)
        for lista in matrix:    #per ogni frame del video
            writer.writerow(lista)


def shape_to_np(shape, dtype="int"):
    """
    Restituisce le una lista contente le coppie
    di coordinate che rappresentano i landmark
    """
    coords = np.zeros((68, 2), dtype=dtype)  #inizializza la lista
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def create_dataset_csv(file_to_create):
  os.chdir(path_git)
  df = pd.read_csv(filename+'_'+file_to_create)
  os.chdir(os.path.join(path_drive, dataset_dir))
  if not os.path.isdir('landmark_csv_'+file_to_create):
    os.makedirs('landmark_csv_'+file_to_create)
    for index in tqdm(range(df.shape[0])):
      videoFile = df['video_name'][index]
      cap= cv2.VideoCapture(videoFile)       #apre il video
      distanceMatrixExt = []          #ConterrÃ  N array ognuno contenente le distanze euclidee per ogni singolo frame
      while(cap.isOpened()):          #Fin quando il video non sarÃ  concluso
        ret, image = cap.read()     #Salva ogni frame in image 

        if ret == False:
          print("errore")
          break            
      
        rects = detector(image, 1)      #Estrae i rettangoli contenenti visi      

        for rect in rects:      #Per ogni rettangolo contenente un viso
          shape = predictor(image, rect)    #Determina i landmark del viso
          shape = shape_to_np(shape)        #Converte i landmark in coordinate (x, y) in un array NumPy              
          
          i = 1
          distanceMatrix = []     #Arrau delle distanze euclidee per i singoli frame
          (x, y, w, h) = cv2.boundingRect(np.array([shape[FACIAL_LANDMARKS_IDXS["mouth"][0]:FACIAL_LANDMARKS_IDXS["mouth"][1]]])) #Estrae i punti per il rettangolo contenente le labbra
          roi = image[y-10:y + h +10, x-10:x + w + 10]   #Estrae il rettangolo contenente le labbra(Con dimensione 10 in piÃ¹ da ogni lato)
          new = cv2.resize(roi, dsize=SIZE, interpolation=cv2.INTER_CUBIC)    #DÃ  al frame dimensione SIZE
            
          xm,ym = FACIAL_LANDMARKS_IDXS["mouth_extern"]  #Prende solo i landmark per le labbra
          print(xm)
          new_shape = []
          new_copy = np.copy(new)     #copia l'immagine
          for (xa, ya) in shape[xm:ym]:       #per ogni coppia di coordinate scelta
              xi = xa - (x-10)                                        # In queste 4 righe di codice prende i landmark 
              yi = ya - (y-10)                                        # dell'immagine originale e li scala in modo da 
              new_x = int((xi * SIZE[0]) / (w+20))                    # adattarli alle nuove dimensioni scandite dalla
              new_y = int((yi * SIZE[1]) / (h+20))                    # costante SIZE
              new_shape.append((new_x, new_y))                        #Aggiunte le coordiante dei frame all'array

          i=1
          for (x1, y1) in new_shape:
              for (x2, y2) in new_shape[i:]:
                  distanceMatrix.append(int(np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2]))))   #Stampa le distanze
              i+=1
              print(distanceMatrix)
          distanceMatrixExt.append("Matrice: "+distanceMatrix)
            
      print_csv_file(videoFile.split(".")[0] + "_m", distanceMatrixExt, 'landmark_csv_'+file_to_create)   #Chiama la funzione per stampare la matrice di distanze nell'omonimo file csv       
      cap.release()

create_dataset_csv('train.csv')
  