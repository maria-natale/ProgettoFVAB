import os
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re
import cv2 
import math 
import tqdm

percorsodataset="/content/drive/MyDrive/Casillo&Natale/roba gaetano/dataset"
test="/content/drive/MyDrive/Casillo&Natale/roba gaetano/test"
train="/content/drive/MyDrive/Casillo&Natale/roba gaetano/train"
standard="/content/drive/MyDrive/Casillo&Natale/roba gaetano"

def csv_di_tutte_le_immagini():
  os.chdir(percorsodataset)
  columns = ['video_name', 'language', 'gender', 'over30', 'lan_gen_age']
  df = pd.DataFrame(columns=columns)
  for immagine in tqdm.tqdm(glob('*.jpg')):
    video_name = os.path.basename(immagine)
    items = video_name.split('_')
    new_name = items[0]+'_'+items[1]+'_'+items[2]+'_'+items[3]+'_'+items[4]
    df.loc[-1] = [new_name, items[0], items[1], items[2], items[0]+'_'+items[1]+'_'+items[2]]
    df.index += 1
  duplicateRowsDF = df[df.duplicated(['video_name'])]
  df=df.drop_duplicates(subset=['video_name'])
  df.sort_values(by = ['video_name'], ascending = True)
  duplicateRowsDF = df[df.duplicated(['video_name'])]
  os.chdir(standard)
  df.to_csv("csvtutteleimmagini.csv", index = False)



    


csv_di_tutte_le_immagini()