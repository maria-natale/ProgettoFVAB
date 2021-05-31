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
filename = ''
path_driveALL = '/content/drive/MyDrive/Casillo&Natale/dataset/dataset_all'
path_drive2 = '/content/drive/MyDrive/Casillo&Natale/dataset/DatasetDueLingue'
dataset_dir = ''

LANGUAGES = {
  1:'Italiano',
  2:'Inglese',
  3: 'Tedesco',
  4:'Spagnolo',
  5: 'Olandese',
  6:'Russo',
  7: 'Giapponese'}
LANGUAGES_N = {
    1:0,
    2:1,
    3:2,
    4:3,
    5:4,
    6:5,
    7:6
}


#crea un file csv contenente i nomi di tutti i video (escludendo le ripetizioni per i video da cui sono state estratte pi� sequenze)
def create_csv_file(path_all, path, combination):
  os.chdir(path_all)
  columns = ['video_name', 'language', 'gender', 'over30', 'lan_gen_age']
  df = pd.DataFrame(columns=columns)
  for video in glob(str(combination[0])+'*.avi'): #per ogni video nella directory crea un'entry nel file csv
    video_name = os.path.basename(video)
    items = video_name.split('_')
    new_name = items[0]+'_'+items[1]+'_'+items[2]+'_'+items[3]+'_'+items[4]
    df.loc[-1] = [new_name, items[0], items[1], items[2], items[0]+'_'+items[1]+'_'+items[2]]
    df.index += 1  
  for video in glob(str(combination[1])+'*.avi'): #per ogni video nella directory crea un'entry nel file csv
    video_name = os.path.basename(video)
    items = video_name.split('_')
    new_name = items[0]+'_'+items[1]+'_'+items[2]+'_'+items[3]+'_'+items[4]
    df.loc[-1] = [new_name, items[0], items[1], items[2], items[0]+'_'+items[1]+'_'+items[2]]
    df.index += 1  
  df=df.drop_duplicates(subset=['video_name'])
  df = df.sample(frac = 1)
  os.chdir(path)
  os.mkdir('file_dataset')
  df.to_csv(os.path.join(path, 'file_dataset', filename+'.csv'), index = False)



#crea tre file csv per group (train, validation, test)
def split_data(path_all, path):
  os.chdir(os.path.join(path, 'file_dataset'))
  df = pd.read_csv(filename+'.csv')
  split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.4, random_state = 42) 
  labelencoder = LabelEncoder()
  df['lan_gen_age'] = labelencoder.fit_transform(df['lan_gen_age'])

  #per stratificazione equilibrata usare lan_gen_age. I video del russo sono sbilanciati quindi usare solo language
  for train_index, test_index in split.split(df, df['language']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
  strat_train_set_all = pd.DataFrame(columns = ['video_name', 'language'])
  os.chdir(path_all)
  for i, row in strat_train_set.iterrows(): 
    video_name = row['video_name']
    language = row['language']
    for videoFile in glob(video_name+'*'): #trova tutti i video che iniziano con video_name
      strat_train_set_all.loc[-1] = [videoFile, language]
      strat_train_set_all.index += 1

  split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state = 42) 
  strat_test_set_all = pd.DataFrame(columns = ['video_name','language'])
  strat_validation_set_all = pd.DataFrame(columns = ['video_name', 'language'])
  for validation_index, test_index in split.split(strat_test_set, strat_test_set['language']):
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



#crea due file contenente nomi dei csv di group con le rispettive etichette
#normalizza ciascun file a 350 righe
def split_csvfiles(path_all, path, combination, group):
  os.chdir(os.path.join(path, 'file_dataset'))
  train = pd.read_csv(filename+'_'+group+'.csv')
  train_label = pd.DataFrame(columns = ['video_name', 'language'])
  os.chdir(os.path.join(path))
  if not os.path.isdir(group):
    os.makedirs(group)
    os.chdir(os.path.join(path_all, 'datasetCSV'))
    for filecsv in tqdm(glob(str(combination[0])+"*.csv")):
      name_to_search = filecsv.split('.')[0]+".avi"
      df = pd.read_csv(filecsv)
      df = df.iloc[:350, :]
      df = df.transpose()
      if any(train.video_name == name_to_search):
        df.to_csv(path+'/'+group+'/'+filecsv, index = False)
        train_label.loc[-1] = [filecsv, filecsv.split('_')[0]]
        train_label.index += 1
    for filecsv in tqdm(glob(str(combination[1])+"*.csv")):
      name_to_search = filecsv.split('.')[0]+".avi"
      df = pd.read_csv(filecsv)
      df = df.iloc[:350, :]
      df = df.transpose()
      if any(train.video_name == name_to_search):
        df.to_csv(path+'/'+group+'/'+filecsv, index = False)
        train_label.loc[-1] = [filecsv, filecsv.split('_')[0]]
        train_label.index += 1

    os.chdir(path)
    if not os.path.isdir('csv'):
      os.mkdir('csv')
    os.chdir('csv')
    train_label.to_csv(filename+'_'+group+'_csv.csv', index = False)



#crea un file per ognuna delle features. La riga i-esima di ogni file corrisponder� ai valori della colonna relativa a quella feature nel file i
def union_features(path_all, path, group):
  os.chdir(os.path.join(path, 'csv'))
  df_train_names = pd.read_csv(filename+'_'+group+'_csv.csv')
  os.chdir(os.path.join(path, group))
  train_targets = pd.DataFrame(columns = ['language'])
  columns_350 = [str(i) for i in range(350)]
  df_features = []
  for i in range (0, 66):
    df_features.append(pd.DataFrame(columns = columns_350))
  
  for i, row in tqdm(df_train_names.iterrows()):
    fl = row['video_name']
    df = pd.read_csv(fl)
    df = df.set_axis(columns_350, axis = 1)
    for i, row in df.iterrows():
      j = i%66
      df_features[j].loc[-1] = row
      df_features[j].index+=1
    train_targets.loc[-1] = [fl.split('_')[0]]
    train_targets.index+=1
  
  os.chdir(path)
  if not os.path.isdir('features'):
    os.mkdir('features')
  os.chdir(os.path.join(path, 'features'))
  os.makedirs(group)
  for i in range (0,66):
    df_features[i].to_csv(group+'/feature'+str(i)+'.csv', index= False)
  os.chdir(os.path.join(path, 'csv'))
  train_targets.to_csv(filename+'_'+group+'_targets_1.csv',  index= False)



if __name__ == '__main__':
  combinations = []
  lang = list(LANGUAGES.keys())
  for i in range(len(lang)):
    for j in range(i+1, len(lang)):
      combinations.append((i+1, j+1))
  
  for combination in combinations: 
    os.chdir(path_drive2)
    dataset_dir = 'dataset_'+str(combination[0])+'_'+str(combination[1])
    os.mkdir(dataset_dir)
    path = os.path.join(path_drive2, dataset_dir)
    filename=str(combination[0])+'_'+str(combination[1])
    create_csv_file(path_driveALL, path, combination)
    split_data(path_driveALL, path)
    for group in ['train', 'validation', 'test']:
      split_csvfiles(path_driveALL, path, combination, group)
      union_features(path_driveALL, path, group)


