import config as cfg
import os
import pandas as pd
import numpy as np
import glob
import json
import functions_get_csv_4_analysis as fct
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# create result folder if doesn't exists
if not os.path.exists(cfg.DL_CSV_output_path):
    os.mkdir(cfg.DL_CSV_output_path)

#list of selected subjects for export
on_subs = cfg.all_subjects  # could be cfg.all_subjects or other
# input data path (interpolate data are on the same path as 'classic' (basic) data)
path_to_folder = cfg.loader_output_path
data_name_ending = '_data_interpolate.csv'  # could be '_data.csv'
event_name_endind  = '_signature.json'

concat_ALL_sub_data = pd.DataFrame()

#loop over subject list
for sub in on_subs:

    #loop over all CSV (for one sub)
    sub_CSVs_path = path_to_folder + sub + '/'
    cpt = 0

    concat_sub_data = pd.DataFrame() #create empty dataframe to concat all subject data

    print('sub_CSVs_path + data_name_ending : ', sub_CSVs_path + data_name_ending)

    for filename in os.listdir(sub_CSVs_path):  # Attention ! avec ça y a rien qui certifie l'ordre des données...  #TODOALBRU : investigate
    #list_of_files = sorted(filter(os.path.isfile, glob.glob(sub_CSVs_path + '*' + data_name_ending))) #Alors qu'ici oui!
    #for path in list_of_files:

        if filename.endswith(data_name_ending):

            #current workind data and event (one csv for one subject)
            data = pd.read_csv(sub_CSVs_path + filename)

            # Data relative body point #
            # "Substract the value of every point of the body by the position of the center point (pose1 for body & _30 for face)
            # loop over center points defined in config
            for count, centerpoint in enumerate(cfg.all_center_points):

                data[cfg.column_2_normalize[count]] = data[cfg.column_2_normalize[count]].values - data[centerpoint].values[:,None] #Broadcasting Pandas<->Numpy

            # Normalization by body size #
            body_size = np.sqrt((data[cfg.waist_x] - data[cfg.pose_center_x])**2 + (data[cfg.waist_y] - data[cfg.pose_center_y])**2)
            #TODO BRU : Moyenne de body_size plutôt qu'un calcul pour chaque ligne ?
            print('body_size : ', body_size)

            if body_size.all():
                data[cfg.pose] = data[cfg.pose] / body_size.values[:,None]
                data[cfg.face] = data[cfg.face] / body_size.values[:,None]
            else:
                print('Warning ! Body size has empty element')

            # Export concatenated data in one excel file #

            # Select data you want : for details --> look in config.py
            part= [cfg.pose, cfg.face, cfg.gaze_angle, cfg.head_orientation, cfg.head_position, cfg.AUr, cfg.AUc]

            flat_part = [item for sublist in part for item in sublist] #flatten list of list

            #add ['EventCode','Presence','Emotion']
            for col_name in ['EventCode','Presence','Emotion']:
                flat_part.append(col_name)

            # set event code
            code = fct.get_Event_code(filename)
            data['EventCode'] = code

            # Set 'Presence' and 'Emotion'
            # Opening JSON file
            print(sub_CSVs_path + filename.replace(data_name_ending, event_name_endind))
            f = open(sub_CSVs_path + filename.replace(data_name_ending, event_name_endind))
            data_json = json.load(f)
            data['Presence'] = data_json['presence']
            data['Emotion'] = data_json['emotion']

            concat_sub_data = concat_sub_data.append(data[flat_part])

    #export in one CSV = one subject
    concat_sub_data.to_csv(cfg.DL_CSV_output_path + '/' + sub + '_positions_x_y.csv', index=False)

    #put every subject data in one table
    # concat_ALL_sub_data = concat_ALL_sub_data.append(concat_sub_data, ignore_index=True)

#Export all data in one excel file
# concat_ALL_sub_data.to_csv(cfg.DL_CSV_output_path + '/' + 'csv_4_DL_RN.csv', index=False)

#Load functions to get csv for deep learning analyses, details on functions in functions_get_csv_4_analysis.py
df = fct.adapt_csv_format(label='Both', normalize_scales=True, write=True)
module_df = fct.module_calcul(df, write=True)
fct.diff_module_calcul(module_df, write=True)