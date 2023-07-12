# Tracking_behavior_IA_2023_BMichelot

If you want to use the scripts or need more information about their functioning, please contact the corresponding author: bruno.michelot@etu.univ-lyon1.fr.

This repository is divided into different "modules" or scripts that allow performing the transformations indicated in Figure 2, "Data Preprocessing Steps." These modules take as input the data extracted by the two open-source software tools, "OpenFace" and "OpenPose." For more information, please visit the following links:

OpenFace: https://github.com/TadasBaltrusaitis/OpenFace/wiki

OpenPose: https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_00_index.html.

Test data to check that the scripts are running properly can be found at the following link and in the "Examples" folder: https://zenodo.org/record/8134430

### 01_script_loader : 

Data restructuring & filtering/smoothing script. 
Converts the .json files supplied by OpenPose (functions in the convert_json.py script) and concatenates them with the .csv supplied by OpenFace and the .txt results of subjective scales (SliderRes file) into a single spreadsheet with a common, easy-to-handle format. 
The script can also be used to filter/smooth OpenPose values for greater precision in extracted data (filter.py functions). 
The script also allows data to be sliced according to conditions, thanks to the EventCodes (A= SelfStim, B= CtrlStim, C= SelfRest, D= CtrlRest, E= SelfSoc, F= CtrlSoc) provided in the .tps file.
The various functions used by the loader are contained and documented in the "functions_loader.py" script. 

This script will produce 36 files .csv per subject, associated with the 36 conditions of the experiment (2 blocks x 3 situations x 6 repetitions). 

##### To run this script, you need to input what is contained in the "Example_Data_To_Preprocess" folder. The paths to the data are entered in the "config.py" script. 

### 02_script_interpolate : 

Data interpolation script. This script takes as input the files obtained with the loader script in the previous step (36 files) and interpolates the bad detections (annotated 0). 
This script produces 36 .csv files similar to the previous ones, with the interpolation and the notation interpolate.csv
This script can be run on a single subject (as with example data) or on several subjects.

### 03_script_get_csv_4_analysis : 

Data normalization & transformation script.
