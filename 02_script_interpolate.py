import numpy as np
import pandas as pd
from pathlib import Path
import config as cfg

"""this script take as input the ouput of 01_script_loader. this script is used for interpolation of bad channels and bad data"""
"""Module : Interpolation"""

repo = cfg.loader_output_path
subject = "CHE324" #subject name to load

#file = "C:/Users/your_path/CHE324/CHE324_9_A_data.csv" #if just one file to interpolate

#for file in Path(repo).glob('**/*data.csv'):
for file in Path(repo).glob(subject +'/*_data.csv'):

    print('Working on : ', file)

    data = pd.read_csv(file)

    for col in data.columns:
        if ('AU' in col) or ('pose11' in col) or ('pose14' in col) or ('face_id' in col) or ('pose10' in col) or ('pose13' in col):
            continue
        elif col=='time':
            continue
        elif 'MAL313' in str(file) and (('pose4' in col) or ('pose7' in col) or ('righthand' in col) or ('lefthand' in col)):
            continue
        else:
            data[col].mask(data[col] == 0, np.nan, inplace=True)
            #multiple = 5
            #data[col].mask(data[col] > data[col].median() + multiple*data[col].std()) | (data[col] < data[col].median() - multiple*data[col].std(), np.nan, inplace=True)
            #print(data[col])


    data.interpolate(axis="index", method="linear",  inplace=True)
    data.interpolate(axis="index", method="linear",  inplace=True, limit_direction='backward')

    print(data.isnull().values.any())
    for col in data.columns:
        nanloc = list(np.where(data[col].isnull())[0])
        print('col : ', col)
        print('nanloc  : ', nanloc)

    new_csv = str(file).replace(".csv", "_interpolate.csv")
    data.to_csv(new_csv, index=False)
