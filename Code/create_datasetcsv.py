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


#unisce tutti i file csv in due file train e test
def file_union():
  os.chdir(os.path.join(path_drive, dataset_dir, 'train_csv'))
  df_train = pd.concat(pd.read_csv(fl) for fl in glob.glob("*.csv"))
  #df_test = []
  os.chdir(os.path.join(path_drive, dataset_dir, 'test_csv'))
  #for file in tqdm(glob.glob("*.csv")):
  #  df_test.append(pd.read_csv(file))  
  df_test = pd.concat(pd.read_csv(fl) for fl in glob.glob("*.csv"))
  os.chdir(os.path.join(path_git, 'file_dataset', 'csv'))
  df_train.to_csv(filename+"_all_train.csv", index = False)
  df_test.to_csv(filename+"_all_test.csv", index = False)

#per ogni riga dei file di train e di test crea un'etichetta target in due file 
def create_targets_file():
  os.chdir(os.path.join(path_drive, dataset_dir, 'train_csv'))
  if not os.path.isdir('labels'):
    os.makedirs('labels')
  train_targets = pd.DataFrame(columns = ['language'])
  for file in tqdm(glob.glob("*.csv")):
    df = pd.read_csv(file)
    for row in df.iterrows():
        train_targets.loc[-1] = [file.split('_')[0]]
        train_targets.index += 1
  train_targets['language'] = train_targets['language'].map(LANGUAGES)
  
  os.chdir(os.path.join(path_drive, dataset_dir, 'test_csv'))
  if not os.path.isdir('labels'):
    os.makedirs('labels')
  test_targets = pd.DataFrame(columns = ['language'])
  for file in tqdm(glob.glob("*.csv")):
    df = pd.read_csv(file)
    for row in df.iterrows():
        test_targets.loc[-1] = [file.split('_')[0]]
        test_targets.index += 1
  train_targets['language'] = train_targets['language'].map(LANGUAGES)
  os.chdir(os.path.join(path_git, 'file_dataset', 'csv'))
  train_targets.to_csv(filename+"targets_all_train.csv", index = False)
  test_targets.to_csv(filename+"targets_all_test.csv", index = False)
      



if __name__ == '__main__':
  #create_folders()
  file_union()
  create_targets_file()


if __name__ == '__main__':
  create_folders()


