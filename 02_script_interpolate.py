import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import config as cfg

repo = cfg.loader_output_path
subject = "CHE324"

#file = "C:/Users/Bruno/Documents/MetaDossier/CHE324/CHE324_9_A_data.csv"

#for file in Path(repo).glob('**/*data.csv'):
for file in Path(repo).glob(subject +'/*_data.csv'):

    print('Working on : ', file)
    # print('Working on : ', type(str(file)))

    data = pd.read_csv(file)
    #print(data)
    #sns.lineplot(data= data['pose10x'], color='r')


    for col in data.columns:
        if ('AU' in col) or ('pose11' in col) or ('pose14' in col) or ('face_id' in col) or ('pose10' in col) or ('pose13' in col):
            continue
        elif col=='time':
            continue
        elif 'MAL313' in str(file) and (('pose4' in col) or ('pose7' in col) or ('righthand' in col) or ('lefthand' in col)):
            continue
        else:
            #print(col)
            #print(data[col])
            #print("Stdev : ", data[col].std)
            #print("median : ", data[col].median())
            data[col].mask(data[col] == 0, np.nan, inplace=True)
            #multiple = 5
            #data[col].mask(data[col] > data[col].median() + multiple*data[col].std()) | (data[col] < data[col].median() - multiple*data[col].std(), np.nan, inplace=True)
            #print(data[col])


    data.interpolate(axis="index", method="linear",  inplace=True)
    #sns.lineplot(data=data['pose10x'], color='g')
    data.interpolate(axis="index", method="linear",  inplace=True, limit_direction='backward')

    #print('Data interpolees : ', data)
    #sns.lineplot(data=data['pose10x'], color='b')
    #plt.show()

    #nanloc = data.loc[pd.isna(data).any(1), :].index
    #print('nanloc : ', nanloc)
    print(data.isnull().values.any())
    #print(data.isnull().sum())
    for col in data.columns:
        nanloc = list(np.where(data[col].isnull())[0])
        print('col : ', col)
        print('nanloc  : ', nanloc)

    #print(type(file))
    new_csv = str(file).replace(".csv", "_interpolate.csv")
    data.to_csv(new_csv, index=False)
