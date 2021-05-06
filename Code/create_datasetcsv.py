import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re
import cv2 

path_git = '/content/ProgettoFVAB'
filename = 'spagnolo_giapponese'
path_drive = '/content/drive/MyDrive/Casillo&Natale'
dataset_dir = 'dataset_Spagnolo_giapponese/datasetCSV'

LANGUAGES = {4:'Spagnolo',
  7: 'Giapponese'}


def create_folders():
  os.chdir(os.path.join(path_git, 'file_dataset'))
  train = pd.read_csv(filename+'_train.csv')
  os.chdir(os.path.join(path_drive, dataset_dir))
  if not os.path.isdir('train_csv'):
    os.makedirs('train_csv')
    os.makedirs('test_csv')
    train_label = pd.DataFrame(columns = ['video_name', 'language'])
    test = pd.DataFrame(columns = ['video_name'])
    for filecsv in glob.glob("*.csv"):
      name_to_search = filecsv.split('.')[0]+ '.avi'
      df = pd.read_csv(filecsv)
      if any(train.video_name == name_to_search):
        df.to_csv('train_csv/'+filecsv, index = False)
        train_label.loc[-1] = [filecsv, filecsv.split('_')[0]]
        train_label.index += 1
      else:
        df.to_csv('test_csv/'+filecsv, index = False)
        test.loc[-1] = [filecsv]
        test.index += 1

    train_label['language'] = train_label['language'].map(LANGUAGES)
    os.chdir(os.path.join(path_git, 'file_dataset/file_csv'))
    train_label.sort_values(by = ['video_name'], ascending = True)
    train_label.to_csv(filename+'_train_csv.csv', index = False)
    test.sort_values(by = ['video_name'], ascending = True)
    test.to_csv(filename+'_test_csv.csv', index = False)


if __name__ == '__main__':
  create_folders()


