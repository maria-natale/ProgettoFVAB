import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

path_git = '/content/ProgettoFVAB'
filename = 'spagnolo_giapponese.csv'
path_drive = '/content/drive/MyDrive/Casillo&Natale'
dataset_dir = 'dataset_Spagnolo_giapponese'


def create_csv_file(path):
  os.chdir(os.path.join(path_drive, dataset_dir))
  columns = ['video_name', 'language', 'gender', 'over30', 'lan_gen_age']
  df = pd.DataFrame(columns=columns)
  for video in glob.glob('*.avi'): #per ogni video nella directory crea un'entry nel file csv
    video_name = os.path.basename(video)
    items = video_name.split('_')
    new_name = items[0]+'_'+items[1]+'_'+items[2]+'_'+items[3]+'_'+items[4]
    df.loc[-1] = [new_name, items[0], items[1], items[2], items[0]+'_'+items[1]+'_'+items[2]]
    df.index += 1
  df = df.sort_index()
  df.drop_duplicates()
  df.to_csv(os.path.join(path_git, path))


def create_train_test():
  os.chdir(os.path.join(path_git, 'file_dataset'))
  df = pd.read_csv(filename)
  split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42) 
  labelencoder = LabelEncoder()
  df['lan_gen_age'] = labelencoder.fit_transform(df['lan_gen_age'])
  print(df)
  for train_index, test_index in split.split(df, df['lan_gen_age']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
  strat_test_set = strat_test_set["video_name"]
  strat_train_set = strat_train_set["video_name"]
  strat_train_set.to_csv('spagnolo_giapponese_train.csv')
  strat_test_set.to_csv('spagnolo_giapponese_test.csv')


if __name__ == '__main__':
  os.chdir(path_git)
  print(os.path.join('file_dataset', filename))
  if not os.path.exists(os.path.join('file_dataset', filename)):
    create_csv_file(os.path.join('file_dataset', filename))
  create_train_test()
   
  
  


