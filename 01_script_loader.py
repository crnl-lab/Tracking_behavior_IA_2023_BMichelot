import config as cfg
import functions_loader as fct

import os



"""to load the data please specify the 4 differents paths explained below and then run the script
This will create two files for every event of a given subject in the output folder. The first file contains every feature for every frame and the second one
describes data for the whole event like its code or the slider response of the subject"""


subject = 'CHE324'
name_file = '2022-03-14_10h38m23s_CodeSujet=CHE324_V=2.csv'
name_video = "2022-03-14_10h38m23s_CodeSujet=CHE324_V=2.avi"

# path to the .csv FILE output given by OpenFace
#path_face_csvfile = "D:\Bruno\processed\DUR306\DUR306_2022-02-11_13h45m23s_V=2.csv"
#path_face_csvfile = "D:/Bruno/processed/PRP307/2022-02-16_15h12m23s_CodeSujet=PRP307_V=2.csv"
path_face_csvfile = cfg.data_processed_path + subject + "/" + name_file

# path to the FOLDER containing all the .json files given by OpenPose
###CAREFUL to end the path with an additional /
#path_pose_folder = "D:\Bruno\processed\DUR306\Openpose_processed/"
path_pose_folder = cfg.data_processed_path + subject + "/Openpose_processed/"

# path to a folder containing:
# the Timestamps .syncIN file given by the camera acquisition application
# the .tps file of the video given by the camera acquisition application CAREFULL changed must be made to the scripts if the video analysed is no longer the 3rd
# the .txt file containing the results of the sliders given by Presentation
#path_tps_folder = "D:\Bruno\Raw\DUR306/"
path_tps_folder = cfg.data_raw_path + subject + "/"

# must be empty folder named as subject name
#output_path = "C:/Users/Bruno\Documents/MetaDossier/DUR306/"
output_path = cfg.loader_output_path + subject + "/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
print("Loading " + output_path[len(output_path) - 7:len(output_path) - 1])


data, timestamps, subject_name = fct.loader(path_face_csvfile, path_pose_folder, path_tps_folder, output_path, filter= True, write=True)


# #TODOBRUAL : ici on pourrait sauvegarder les infos precedentes et pouvoir executer le code suivant direct,
# sans repasser par le loader
print("cutting video")
video_path = cfg.data_raw_path + subject + "/" + name_video
fct.cut_video(video_path, timestamps, output_path, subject_name)
