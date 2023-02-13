import pandas as pd
import os
import glob
import time
import config as cfg


def adapt_csv_format(label):

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
        print("--- %s seconds ---" % (time.time() - start_time))
        Global_csv = pd.concat(data, ignore_index=True)

    moving_columns = Global_csv.pop('EventCode')
    Global_csv.insert(0, 'EventCode', moving_columns)
    moving_columns = Global_csv.pop('Subject')
    Global_csv.insert(0, 'Subject', moving_columns)

    Global_csv = Global_csv.rename({'EventCode': 'Condition'}, axis=1)

    if label == 'Condition':
        Global_csv = Global_csv.drop(columns=['Presence', 'Emotion'])
    elif label == 'Subjectivity':
        moving_columns = Global_csv.pop('Presence')
        Global_csv.insert(1, 'Presence', moving_columns)
        moving_columns = Global_csv.pop('Emotion')
        Global_csv.insert(1, 'Emotion', moving_columns)
        Global_csv = Global_csv.drop(columns=['Condition'])

    Global_csv.to_csv(cfg.DL_CSV_output_path + '/' + 'All_Subs_Positions.csv',
                      index=False, header=True)