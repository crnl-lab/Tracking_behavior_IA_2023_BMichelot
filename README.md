# Tracking_behavior_IA_2023_BMichelot

If you want to use the scripts or need more information about their functioning, please contact the corresponding author: bruno.michelot@etu.univ-lyon1.fr.

This repository is divided into different "modules" or scripts that allow performing the transformations indicated in Figure 2, "Data Preprocessing Steps." These modules take as input the data extracted by the two open-source software tools, "OpenFace" and "OpenPose." For more information, please visit the following links:

OpenFace: https://github.com/TadasBaltrusaitis/OpenFace/wiki

OpenPose: https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_00_index.html.

Test data to check that the scripts are running properly can be found at the following link and in the "Examples" folder: https://zenodo.org/record/8138258

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
This script takes input files obtained after script loader or script interpolate.
This script normalizes the data with respect to a relative point and subject size. 
It also generates a table suitable for machine learning algorithms, labeling the conditions (changing EventCode from A, B, C, D, E, F to 65, 66, 67, 68, 69, 70) and taking features of our choice that we wish to give to the models (to define in config.py). It can also be used to add results from subjective scales, normalizing them to min/max. 
Finally, it transforms position data into Euclidean distances (modulus) or differences in Euclidean distances (modulus differences). 
(except normalizations, all functions are in functions_get_csv_4_analysis.py). 

### XGBoost.py : 

XGBoost model script in jupyter notebook format. This script takes as input the .csv table obtained after script_get_csv_4_analysis. 
The contrast/classification of interest can be selected according to eventcode (65 = SelfStim, 66 = CtrlStim, 67 = SelfRest, 68 = CtrlRest, 69 = SelfSoc, 70 = CtrlSoc). 
N-fold cross validation is performed, where N is the number of subjects
It also allows to obtain feature importances for each classification and save them, as is the case with accuracy. 

### LSTM folder : 

#### 1) train.py <dataset.csv>

This script trains our LSTM model on the dataset obtained after script_get_csv_4_analysis. 
N-fold cross validation is performed, where N is the number of subjects, i.e. we will have one model per test subject (train on all other subjects) and 3 runs for each fold.
The target classes with their labels are given as two lists in the code (lines 59-60). 
Example:
list_targets = [0, 3]
list_labels = [0, 1]
This trains a binary classifier with class number 0 and class number 3 and assigns labels 0 and 1 to them.
The class indexes are: (0: SelfStim  1: CtrlStim  2: SelfRest  3: CtrlRest  4: SelfSoc  5: CtrlSoc)

#### 2) testdir.py <input_dir> <explain>

This script evaluates the models trained with train.py and stored in <input_dir>. It computes an average accuracy. 
If <explain> is 1, feature attributions (with Integrated Gradients) will be computed and averaged over all time steps and runs and stored in "feature_attributions.csv".
