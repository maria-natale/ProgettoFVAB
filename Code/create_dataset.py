import os
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import re
import cv2 
import math 

path_git = '/content/ProgettoFVAB'
filename = 'spagnolo_giapponese'
path_drive = '/content/drive/MyDrive/Casillo&Natale'
dataset_dir = 'dataset_Spagnolo_giapponese'

LANGUAGES = {4:'Spagnolo',
  7: 'Giapponese'}


#crea un file csv contenente i nomi di tutti i video (escludendo le ripetizioni per i video da cui sono state estratte più sequenze)
def create_csv_file(path):
  os.chdir(os.path.join(path_drive, dataset_dir))
  columns = ['video_name', 'language', 'gender', 'over30', 'lan_gen_age']
  df = pd.DataFrame(columns=columns)
  for video in glob('*.avi'): #per ogni video nella directory crea un'entry nel file csv
    video_name = os.path.basename(video)
    items = video_name.split('_')
    new_name = items[0]+'_'+items[1]+'_'+items[2]+'_'+items[3]+'_'+items[4]
    df.loc[-1] = [new_name, items[0], items[1], items[2], items[0]+'_'+items[1]+'_'+items[2]]
    df.index += 1
  duplicateRowsDF = df[df.duplicated(['video_name'])]
  print(duplicateRowsDF)
  print(len(df))  
  df=df.drop_duplicates(subset=['video_name'])
  df.sort_values(by = ['video_name'], ascending = True)
  duplicateRowsDF = df[df.duplicated(['video_name'])]
  print(df)
  print(len(df))
  df.to_csv(os.path.join(path_git, path), index = False)


#crea due file csv, uno contenente i nomi dei video di train e i rispettivi tag e un altro contenente i nomi dei video di test
def create_train_test():
  os.chdir(os.path.join(path_git, 'file_dataset'))
  df = pd.read_csv(filename+'.csv')
  split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42) 
  labelencoder = LabelEncoder()
  df['lan_gen_age'] = labelencoder.fit_transform(df['lan_gen_age'])
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
  strat_test_set_all = pd.DataFrame(columns = ['video_name'])
  for i, row in strat_test_set.iterrows():
    video_name = row['video_name']
    for videoFile in glob(video_name+'*'): #trova tutti i video che iniziano con video_name
      strat_test_set_all.loc[-1] = [videoFile]
      strat_test_set_all.index += 1  
  strat_train_set_all['language'] = strat_train_set_all['language'].map(LANGUAGES) #trasforma il numero della lingua nella stringa corrispondente
  os.chdir(os.path.join(path_git, 'file_dataset'))
  strat_test_set.sort_values(by = ['video_name'], ascending = True)
  strat_train_set.sort_values(by = ['video_name'], ascending = True)
  strat_train_set_all.to_csv(filename+'_train.csv',index = False)
  strat_test_set_all.to_csv(filename+'_test.csv',index = False)


#salva i frame per i video di training nella cartella train
def save_frames_train_test():
  os.chdir(os.path.join(path_git, 'file_dataset'))
  train = pd.read_csv(filename+'_train.csv')
  os.chdir(os.path.join(path_drive, dataset_dir))
  if not os.path.isdir('train'):
    os.makedirs('train')
    for i in tqdm(range(train.shape[0])):
      video_name = train['video_name'][i]
      count = 0
      cap = cv2.VideoCapture(video_name)   #prende il video dal path
      frameRate = cap.get(5) #frame rate
      x=1
      while(cap.isOpened()):
        frameId = cap.get(1) #prende il frame corrente
        ret, frame = cap.read()
        if (ret != True):
          break
        file_tosave ='train/' + video_name+"_frame%d.jpg" % count            
        count += 1 
        cv2.imwrite(file_tosave, frame)
      cap.release()

  os.chdir(os.path.join(path_git, 'file_dataset'))
  test = pd.read_csv(filename+'_test.csv')
  os.chdir(os.path.join(path_drive, dataset_dir))
  if not os.path.isdir('test'):
    os.makedirs('test')
    for i in tqdm(range(test.shape[0])):
      video_name = test['video_name'][i]
      count = 0
      cap = cv2.VideoCapture(video_name)   #prende il video dal path
      frameRate = cap.get(5) #frame rate
      x=1
      while(cap.isOpened()):
        frameId = cap.get(1) #prende il frame corrente
        ret, frame = cap.read()
        if (ret != True):
          break
        file_tosave ='test/' + video_name+"_frame%d.jpg" % count            
        count += 1 
        cv2.imwrite(file_tosave, frame)
      cap.release()


#csv image, imagini con etichetta
def etichetta_immagine():
  images = glob(os.path.join(path_drive, dataset_dir+"/train/*.jpg"))
  train_image = []
  train_class = []
  for i in tqdm(range(len(images))):
    # nome immagini
    train_image.append(images[i].split('/')[7])
    # classe immagini 
    train_class.append(LANGUAGES[int(images[i].split('/')[7].split('_')[0])])
# immagini e classe in df
  train_data = pd.DataFrame()
  train_data['frame'] = train_image
  train_data['language'] = train_class

  images = glob(os.path.join(path_drive, dataset_dir+"/test/*.jpg"))
  test_image = []
  test_class = []
  for i in tqdm(range(len(images))):
    # nome immagini
    test_image.append(images[i].split('/')[7])
    # classe immagini 
    test_class.append(LANGUAGES[int(images[i].split('/')[7].split('_')[0])])
# immagini e classe in df
  test_data = pd.DataFrame()
  test_data['frame'] = test_image
  test_data['language'] = test_class

# dataframe in csv
  train_data.to_csv(os.path.join(path_drive, dataset_dir, 'etichette', filename+'ImmaginiClasse_train.csv'),header=True, index=False)
  test_data.to_csv(os.path.join(path_drive, dataset_dir, 'etichette', filename+'ImmaginiClasse_test.csv'),header=True, index=False)


if __name__ == '__main__':
  os.chdir(path_git)
  print(os.path.join('file_dataset', filename+'.csv'))
  #if not os.path.exists(os.path.join('file_dataset', filename+'.csv')):
  #create_csv_file(os.path.join('file_dataset', filename+'.csv'))
  #create_train_test()
  save_frames_train_test()
  #etichetta_immagine()
   
  
  


