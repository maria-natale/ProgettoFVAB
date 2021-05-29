import os
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re
import cv2 
import math 
import shutil


path_git = '/content/ProgettoFVAB'
filename = '4_7'
path_drive = '/content/drive/MyDrive/Casillo&Natale/dataset'
dataset_dir = 'dataset_4_7'

"""LANGUAGES = {
  1:'Italiano',
  2:'Inglese',
  3: 'Tedesco',
  4:'Spagnolo',
  5: 'Olandese',
  6:'Russo',
  7: 'Giapponese'}"""
LANGUAGES = {
  4:'Spagnolo',
  7:'Giapponese'
}
LANGUAGES_N = {
  4:0,
  7:1
}

#crea un file csv contenente i nomi di tutti i video (escludendo le ripetizioni per i video da cui sono state estratte più sequenze)
def create_csv_file(path):
  os.chdir(path)
  columns = ['video_name', 'language', 'gender', 'over30', 'lan_gen_age']
  df = pd.DataFrame(columns=columns)
  for video in glob('*.avi'): #per ogni video nella directory crea un'entry nel file csv
    video_name = os.path.basename(video)
    items = video_name.split('_')
    new_name = items[0]+'_'+items[1]+'_'+items[2]+'_'+items[3]+'_'+items[4]
    df.loc[-1] = [new_name, items[0], items[1], items[2], items[0]+'_'+items[1]+'_'+items[2]]
    df.index += 1  
  df=df.drop_duplicates(subset=['video_name'])
  df = df.sample(frac = 1)
  df.to_csv(os.path.join(path, 'file_dataset', filename+'.csv'), index = False)


#crea tre file csv, train, test e validation
def split_data(path):
  os.chdir(os.path.join(path, 'file_dataset'))
  df = pd.read_csv(filename+'.csv')
  split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.4, random_state = 42) 
  labelencoder = LabelEncoder()
  df['lan_gen_age'] = labelencoder.fit_transform(df['lan_gen_age'])

  #per stratificazione equilibrata usare lan_gen_age. I video del russo sono sbilanciati quindi usare solo language
  for train_index, test_index in split.split(df, df['lan_gen_age']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
  strat_train_set_all = pd.DataFrame(columns = ['video_name', 'language'])
  os.chdir(os.path.join(path_drive, dataset_dir))
  for i, row in strat_train_set.iterrows(): 
    video_name = row['video_name']
    language = row['language']
    for videoFile in glob(video_name+'*'): #trova tutti i video che iniziano con video_name
      strat_train_set_all.loc[-1] = [videoFile, language]
      strat_train_set_all.index += 1

  split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state = 42) 
  strat_test_set_all = pd.DataFrame(columns = ['video_name','language'])
  strat_validation_set_all = pd.DataFrame(columns = ['video_name', 'language'])
  for validation_index, test_index in split.split(strat_test_set, strat_test_set['lan_gen_age']):
    strat_validation_set = df.loc[validation_index]
    strat_test_set = df.loc[test_index]
  
  for i, row in strat_validation_set.iterrows():
    video_name = row['video_name']
    language = row['language']
    for videoFile in glob(video_name+'*'): #trova tutti i video che iniziano con video_name
      strat_validation_set_all.loc[-1] = [videoFile, language]
      strat_validation_set_all.index += 1  
  for i, row in strat_test_set.iterrows():
    video_name = row['video_name']
    language = row['language']
    for videoFile in glob(video_name+'*'): #trova tutti i video che iniziano con video_name
      strat_test_set_all.loc[-1] = [videoFile, language]
      strat_test_set_all.index += 1  
  
  os.chdir(os.path.join(path, 'file_dataset'))
  strat_train_set_all.to_csv(filename+'_train.csv',index = False)
  print(strat_train_set_all.shape)
  print(strat_validation_set_all.shape)
  print(strat_test_set_all.shape)
  strat_validation_set_all.to_csv(filename+'_validation.csv',index = False)
  strat_test_set_all.to_csv(filename+'_test.csv',index = False)


def copy_video(path, group):
  os.chdir(os.path.join(path, 'file_dataset'))
  train = pd.read_csv(filename+'_'+group+'.csv')
  os.chdir(os.path.join(path))
  if not os.path.isdir(group):
    os.makedirs(group)
    for classe in LANGUAGES_N.keys():
      os.makedirs(group+'/'+str(LANGUAGES_N[classe]))
    for i in tqdm(range(train.shape[0])):
      video_name = train['video_name'][i]
      lan = video_name.split('_')[0]
      shutil.copy(video_name, group+'/'+str(LANGUAGES_N[int(lan)])+'/'+video_name)
  

if __name__ == '__main__':
  path = os.path.join(path_drive, dataset_dir)
  #create_csv_file(path)
  #split_data(path)
  #copy_video(path, 'train')
  copy_video(path, 'validation')
  copy_video(path, 'test')
