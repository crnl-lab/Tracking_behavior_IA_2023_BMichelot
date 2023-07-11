import config as cfg
import functions_loader as fct
import os

"""to load the data please specify the 4 differents paths explained below and then run the script
This will create two files for every event of a given subject in the output folder. The first file contains every feature for every frame and the second one
describes data for the whole event like its code or the slider response of the subject"""
"""Module : Restructuring --> see convert_json and functions_loader.py & Filtering/Smoothing --> see filter.py and functions_loader.py (fct: loader)"""


subject = 'CHE324' #subject name to load
name_file = '2022-03-14_10h38m23s_CodeSujet=CHE324_V=2.csv' #csv of the subject (OpenFace)
#name_video = "2022-03-14_10h38m23s_CodeSujet=CHE324_V=2.avi" #avi of the subject (OpenFace)

# path to the .csv FILE output given by OpenFace
path_face_csvfile = cfg.data_processed_path + subject + "/" + name_file

# path to the FOLDER containing all the .json files given by OpenPose
###CAREFUL to end the path with an additional /
path_pose_folder = cfg.data_processed_path + subject + "/Openpose_processed/"

# path to a folder containing:
# the Timestamps .syncIN file given by the camera acquisition application
# the .tps file of the video given by the camera acquisition application
# the .txt file containing the results of the sliders given by Presentation Software
path_tps_folder = cfg.data_raw_path + subject + "/"

# must be empty folder named as subject name
output_path = cfg.loader_output_path + subject + "/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
print("Loading " + output_path[len(output_path) - 7:len(output_path) - 1])


data, timestamps, subject_name = fct.loader(path_face_csvfile, path_pose_folder, path_tps_folder, output_path, filter= True, write=True)

# print("cutting video")
# video_path = cfg.data_raw_path + subject + "/" + name_video
# fct.cut_video(video_path, timestamps, output_path, subject_name)
