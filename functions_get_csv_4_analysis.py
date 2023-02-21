import math
import pandas as pd
import os
import glob
import time
import config as cfg
import warnings


def adapt_csv_format(label, normalize_scales=True, write=True):

    ## Allow to have a new csv table adapted for deep learning analysis with rawest datas (i.e. positions x/y/z).
    ## The format is the same as diff_module files that are commonly used in the project.
    ## Posibility to choose if we want to classify by situations or subjective scales with argument "label"

    start_time = time.time()
    directory = cfg.DL_CSV_output_path
    os.chdir(directory)

    files = []
    subjects = []
    data = {}

    for csv in glob.glob('*x_y.csv'):
        files.append(csv)
    for file in files:
        subjects.append(file.split('_')[0])
        data[file.split('_')[0]] = pd.read_csv(directory + file, sep=',')

    for sub in subjects:
        data[sub]['Subject'] = sub
        for col in data[sub][['Emotion', 'Presence']]:
            if normalize_scales == True:
                data[sub][col] = (data[sub][col] - data[sub][col].min()) / (data[sub][col].max() - data[sub][col].min())
                print("Transformation des données")
                print("--- %s seconds ---" % (time.time() - start_time))
                Global_csv = pd.concat(data, ignore_index=True)

    moving_columns = Global_csv.pop('EventCode')
    Global_csv.insert(0, 'EventCode', moving_columns)
    moving_columns = Global_csv.pop('Subject')
    Global_csv.insert(0, 'Subject', moving_columns)
    moving_columns = Global_csv.pop('Presence')
    Global_csv.insert(2, 'Presence', moving_columns)
    moving_columns = Global_csv.pop('Emotion')
    Global_csv.insert(2, 'Emotion', moving_columns)

    Global_csv = Global_csv.rename({'EventCode': 'Condition'}, axis=1)

    if label == 'Both':
        print('Label choice : Both')
    elif label == 'Condition':
        Global_csv = Global_csv.drop(columns=['Presence', 'Emotion'])
        print('Label choice : Condition')
    elif label == 'Subjectivity':
        Global_csv = Global_csv.drop(columns=['Condition'])
        print('Label choice : Subjectivity')

    if write == True:
        Global_csv.to_csv(cfg.DL_CSV_output_path + '/' + 'All_Subs_Positions.csv',
                      index=False, header=True)
    return Global_csv


def module_calcul(df, write=True):

    ###Function that calculate euclidian distances of every features at every frame (concatenation of pose x/y to
    ###obtain only 1 column pose)

    print("Voici le tableau des données : ", df)

    warnings.simplefilter(action='ignore', category=FutureWarning)

    Modules = []

    for col in df.columns:
        if 'x' in col:
            Modules.insert(len(df.columns), col.replace('x', ''))
        elif 'AU' in col:
            Modules.insert(len(df.columns), col)

    Modules_csv = pd.DataFrame(columns=Modules)

    print("Calcul des modules, cela va prendre du temps (plusieurs heures). C'est le moment pour une pause café bien méritée.")
    for i in range(len(df)):
        temp = []
        for col in Modules_csv.columns:
            if col[0] == '_':
                temp.append(math.sqrt((df['x' + col][i]**2) + (df['y' + col][i]**2)))
            elif 'AU' in col:
                temp.append(df[col][i])
            else:
                temp.append(math.sqrt((df[col + 'x'][i] ** 2) + (df[col + 'y'][i] ** 2)))
        new_line = pd.Series(temp, index=Modules_csv.columns)
        Modules_csv = Modules_csv.append(new_line, ignore_index=True)
    print(Modules_csv)

    columns_label = df.iloc[:,:4]

    Modules_csv = pd.concat([Modules_csv,columns_label], axis=1)

    moving_columns = Modules_csv.pop('Condition')
    Modules_csv.insert(0, 'Condition', moving_columns)
    moving_columns = Modules_csv.pop('Subject')
    Modules_csv.insert(0, 'Subject', moving_columns)
    moving_columns = Modules_csv.pop('Presence')
    Modules_csv.insert(2, 'Presence', moving_columns)
    moving_columns = Modules_csv.pop('Emotion')
    Modules_csv.insert(2, 'Emotion', moving_columns)

    if write == True:
        Modules_csv.to_csv(cfg.DL_CSV_output_path + '/' + 'All_Subs_Modules.csv',
                      index=False, header=True)

    return Modules_csv

def diff_module_calcul(module_df, write=True):

    ###Function that calculate differences of modules obtained, i.e. module x - module x-1 where x is a frame (= a line
    ###in the table)

    warnings.simplefilter(action='ignore', category=FutureWarning)

    module_df = pd.read_csv("C:/Users\Bruno\Documents\Meta_Output\All_Subs_Modules.csv", sep=';')
    Diff_Modules = pd.DataFrame(columns=module_df.columns)
    Diff_Modules = Diff_Modules.drop(['Subject', 'Emotion', 'Presence'], axis=1)

    print("Calcul des différences de modules, cela va prendre du temps (plusieurs heures).")
    for i in range (1,len(module_df)):
        temp = [module_df['Condition'][i]]
        for col in module_df.columns[4::]:
            if 'AU' in col:
                temp.append(module_df[col][i])
            else:
                temp.append(abs(module_df[col][i] - module_df[col][i-1]))
        new_line = pd.Series(temp, index=Diff_Modules.columns)
        Diff_Modules = Diff_Modules.append(new_line, ignore_index=True)

    columns_label = module_df[['Subject', 'Emotion', 'Presence']][:-1]
    Diff_Modules = pd.concat([Diff_Modules, columns_label],axis=1)

    moving_columns = Diff_Modules.pop('Subject')
    Diff_Modules.insert(0, 'Subject', moving_columns)
    moving_columns = Diff_Modules.pop('Presence')
    Diff_Modules.insert(2, 'Presence', moving_columns)
    moving_columns = Diff_Modules.pop('Emotion')
    Diff_Modules.insert(2, 'Emotion', moving_columns)

    print(Diff_Modules)

    if write == True:
        Diff_Modules.to_csv(cfg.DL_CSV_output_path + '/' + 'All_Subs_Diff_Modules.csv',
                      index=False, header=True)

    return Diff_Modules