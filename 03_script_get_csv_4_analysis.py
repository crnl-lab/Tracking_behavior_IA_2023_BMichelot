import config as cfg
import os
import pandas as pd
import numpy as np
import glob

# create result folder if doesn't exists
if not os.path.exists(cfg.median_CSV_output_path):
    os.mkdir(cfg.median_CSV_output_path)

#list of selected subjects for export
on_subs = cfg.some_subs  # could be cfg.all_subjects or other 
# input data path (iterpolate data are on the same path as 'classic' (basic) data)
path_to_folder = cfg.loader_output_path
data_name_ending = '_data_interpolate.csv'  # could be '_data.csv'
event_name_endind  = '_signature.json'
#name_code = 



#loop over subject list
for sub in on_subs:
    
    #loop over all CSV (for one sub)
    sub_CSVs_path = path_to_folder + sub + '/'
    cpt = 0
    
    concat_sub_data = pd.DataFrame() #create empty dataframe to concat all subject data
    
    print('sub_CSVs_path + data_name_ending : ', sub_CSVs_path + data_name_ending)
    
    #for filename in os.listdir(sub_CSVs_path):  # Attention ! avec ça y a rien qui certifie l'ordre des données...  #TODOALBRU : investigate
    list_of_files = sorted(filter(os.path.isfile, glob.glob(sub_CSVs_path + '*' + data_name_ending))) #Alors qu'ici oui!
    print(list_of_files)
    for path in list_of_files:
        
        #filename = 'CHE324_10_C_data_interpolate.csv'
            
        #if filename.endswith(data_name_ending): 
            
        #print('Working on ' + filename)
        print('Working on ' + path)
        
        #current workind data and event (one csv for one subject)
        #data = pd.read_csv(sub_CSVs_path + filename)
        #event = pd.read_csv(sub_CSVs_path + filename.replace(data_name_ending, event_name_endind))
        
        data = pd.read_csv(path)
        event = pd.read_csv(path.replace(data_name_ending, event_name_endind))
        
        #print(data.columns.get_indexer(cfg.all_central_points))
        
        ## Data relative body point
        # "Substract the value of every point of the body by the position of the center point (1)
        # Same thing for the face (point 30) for OpenFace"
        # loop over center points defined in config
        for count, centerpoint in enumerate(cfg.all_center_points):
            
            data[cfg.column_2_normalize[count]] = data[cfg.column_2_normalize[count]].values - data[centerpoint].values[:,None] #Broadcasting Pandas<->Numpy
            # on enlève qq chose qui fait (N) à un truc de (N,8) et donc faut passer en (N,1) et on le fait en numpy donc .values
            #data[cfg.column_2_normalize[count]] = data[cfg.column_2_normalize[count]].sub(data[centerpoint], axis=0)
        
        
        ## Normalization by body size
        cfg.pose
        cfg.face
        
        body_size = np.sqrt((data[cfg.waist_x] - data[cfg.pose_center_x])**2 + (data[cfg.waist_y] - data[cfg.pose_center_y])**2)
        #TODO BRU : je comprend pas pourquoi on a pas qu'une seule valeur de 'bodysize', la on a une colomne entière (calculé pour chaque ligne)
        # parceque genre la body size elle change pas pour un individu non ? on devrait pas faire la moyenne de la colomne bodysize ?
        print('body_size : ', body_size)
        if body_size.all():
            data[cfg.pose] = data[cfg.pose] / body_size.values[:,None]
            data[cfg.face] = data[cfg.face] / body_size.values[:,None]
        else:
            print('Warning ! Body size has empty element')            
        
        
        ## Export concatenated data in one excel file
        
        # Select data you want
        #Part is a list containing the name of all the features you want to add in the file, your choice are :
        #pose : point of the  body
        #face : point of the face (from OpenFace)
        #AU : AU intensite exept for the binary value of AU 28
        #hand : points of both hands
        #gaze : gaze_angle
        #head_orientation : angle of the head in 3D
        #head_position : postion of the head in  3D
        part= [cfg.pose, cfg.face, cfg.hand, cfg.gaze_angle, cfg.gaze, cfg.head_orientation, cfg.head_position, cfg.AU]  
        #TODO : c'est les mêmes 'pose' et 'face' que pour la normalisation,
        #genre c'est la parametres définits dans config 1 bonne fois pour toutes (dans le script de G. ct pas le cas, plusieurs endroits)
        
        flat_part = [item for sublist in part for item in sublist]
        print('flat_part ', flat_part)
            
        concat_sub_data = concat_sub_data.append(data[flat_part])
        #Est-ce qu'il y a bien toutes les colomnes voulues dans le bon ordre ?
            
    concat_sub_data.to_csv(cfg.median_CSV_output_path + '/' + sub + '_Median_Std.csv') # Pourquoi ça s'appelle median std ??
    # export everything in one CSV  # quid de l'ordre ??

