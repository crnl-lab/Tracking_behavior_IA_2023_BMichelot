import os




#Path to Data :
if os.uname().nodename == 'lx37-Kubiak': #'tkz-XPS'
    data_path = "/home/lx37/Projets/FPerrin2022_VideoComaEEG_Bruno/Comportement/Data_Video"
    data_raw_path = data_path + "/raw/"
    data_processed_path = data_path + "/OP_OF_processed/"
    loader_output_path = data_path + "/after_loading/"
    median_CSV_output_path = data_path + "/Post_DLCSV/"
else:
    data_raw_path = "D:/Bruno/Raw/" 
    data_processed_path = "D:/Bruno/processed/"
    loader_output_path =  "C:/Users/Bruno/Documents/MetaDossier/"
    median_CSV_output_path = "Tructruc/Subjects_4_Preplotting/" #TODO BRU


###########################################################################
############# Config params for step 3 : get_CSV_4_analysis ###############
###########################################################################

all_subjects = ['CHE324', 'DES316', 'EDC312', 'GIC326', 'GRL315', 'HAA328', 'MAL313', 'MIL332', 'MOP309', 'PAP319',
                'POA318', 'PRP307', 'REL317', 'ROL311', 'SAE308', 'TAI310']

some_subs = ['CHE324']

### Relative body computation params :

#List of all center points 
all_center_points = ['pose1x', 'pose1y', 'x_30', 'y_30']

# List of values to normalise by center points 
# Beware : the central point will be 0 as it's substracted by itself
# to get his values you would normalize by another way
pose_x =  ['pose0x', 'pose1x', 'pose2x',  'pose3x',  'pose4x', 'pose5x', 'pose6x', 'pose7x', 'pose8x', 'pose15x', 'pose16x', 'pose17x', 'pose18x']
pose_y =  ['pose0y', 'pose1y', 'pose2y', 'pose3y', 'pose4y',  'pose5y',  'pose6y',  'pose7y', 'pose8y', 'pose15y', 'pose16y',  'pose17y', 'pose18y']
#TODO BRU : y a pas x_30 ni y_30 dans la liste ci-dessous alors qu'iels devraient être à 0 non ?
face_x = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67']
face_y = ['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12', 'y_13', 'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 'y_21', 'y_22', 'y_23', 'y_24', 'y_25', 'y_26', 'y_27', 'y_28', 'y_29', 'y_31', 'y_32', 'y_33', 'y_34', 'y_35', 'y_36', 'y_37', 'y_38', 'y_39', 'y_40', 'y_41', 'y_42', 'y_43', 'y_44', 'y_45', 'y_46', 'y_47', 'y_48', 'y_49', 'y_50', 'y_51', 'y_52', 'y_53', 'y_54', 'y_55', 'y_56', 'y_57', 'y_58', 'y_59', 'y_60', 'y_61', 'y_62', 'y_63', 'y_64', 'y_65', 'y_66', 'y_67']
column_2_normalize = [pose_x, pose_y, face_x, face_y] # for looping - should be the same size as 'all_central_points'


## Normalization computation params

#
pose = ['pose0x', 'pose1x', 'pose2x', 'pose3x', 'pose4x', 'pose5x', 'pose6x', 'pose7x', 'pose8x', 'pose15x', 'pose16x',
            'pose17x', 'pose18x',
            'pose0y', 'pose1y', 'pose2y', 'pose3y', 'pose4y', 'pose5y', 'pose6y', 'pose7y', 'pose8y', 'pose15y',
            'pose16y', 'pose17y', 'pose18y'
            ]

#
face = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13',
            'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25',
            'x_26', 'x_27', 'x_28', 'x_29', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37',
            'x_38', 'x_39', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49',
            'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_60', 'x_61',
            'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67',
            'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6', 'y_7', 'y_8', 'y_9', 'y_10', 'y_11', 'y_12', 'y_13',
            'y_14', 'y_15', 'y_16', 'y_17', 'y_18', 'y_19', 'y_20', 'y_21', 'y_22', 'y_23', 'y_24', 'y_25',
            'y_26', 'y_27', 'y_28', 'y_29', 'y_30', 'y_31', 'y_32', 'y_33', 'y_34', 'y_35', 'y_36', 'y_37',
            'y_38', 'y_39', 'y_40', 'y_41', 'y_42', 'y_43', 'y_44', 'y_45', 'y_46', 'y_47', 'y_48', 'y_49',
            'y_50', 'y_51', 'y_52', 'y_53', 'y_54', 'y_55', 'y_56', 'y_57', 'y_58', 'y_59', 'y_60', 'y_61',
            'y_62', 'y_63', 'y_64', 'y_65', 'y_66', 'y_67'
            ]

waist_x = 'pose8x'
waist_y = 'pose8y'
pose_center_x = 'pose1x'
pose_center_y = 'pose1y'

AU = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r',
              'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r' , 'AU28_c']
hand = ['lefthand0x', 'lefthand0y', 'lefthand1x', 'lefthand1y', 'lefthand2x', 'lefthand2y', 'lefthand3x', 'lefthand3y', 'lefthand4x', 'lefthand4y', 'lefthand5x', 'lefthand5y', 'lefthand6x', 'lefthand6y', 'lefthand7x', 'lefthand7y', 'lefthand8x', 'lefthand8y', 'lefthand9x', 'lefthand9y', 'lefthand10x', 'lefthand10y', 'lefthand11x', 'lefthand11y', 'lefthand12x', 'lefthand12y', 'lefthand13x', 'lefthand13y', 'lefthand14x', 'lefthand14y', 'lefthand15x', 'lefthand15y', 'lefthand16x', 'lefthand16y', 'lefthand17x', 'lefthand17y', 'lefthand18x', 'lefthand18y', 'lefthand19x', 'lefthand19y', 'lefthand20x', 'lefthand20y',
                'righthand0x', 'righthand0y', 'righthand1x', 'righthand1y', 'righthand2x', 'righthand2y', 'righthand3x', 'righthand3y', 'righthand4x', 'righthand4y', 'righthand5x', 'righthand5y', 'righthand6x', 'righthand6y', 'righthand7x', 'righthand7y', 'righthand8x', 'righthand8y', 'righthand9x', 'righthand9y', 'righthand10x', 'righthand10y', 'righthand11x', 'righthand11y', 'righthand12x', 'righthand12y', 'righthand13x', 'righthand13y', 'righthand14x', 'righthand14y', 'righthand15x', 'righthand15y', 'righthand16x', 'righthand16y', 'righthand17x', 'righthand17y', 'righthand18x', 'righthand18y', 'righthand19x', 'righthand19y', 'righthand20x', 'righthand20y']
        #vecteur 3D pour chaque oeil
gaze = ['gaze_0_x','gaze_0_y','gaze_0_z','gaze_1_x','gaze_1_y','gaze_1_z']
head_orientation = ['pose_Rx','pose_Ry','pose_Rz']
head_position = ['pose_Tx','pose_Ty','pose_Tz']

#ajouts Bruno
#rightarm = ['pose5x', 'pose5y', 'pose6x', 'pose6y', 'pose7x', 'pose7y']
#leftarm = ['pose2x', 'pose2y', 'pose3x', 'pose3y', 'pose4x', 'pose4y']
#face_single_point = ['x_28', 'y_28']
#mouth = ['x_48', 'y_48', 'x_51', 'y_51', 'x_54', 'y_54', 'x_57', 'y_57', 'x_62', 'y_62', 'x_66', 'y_66']
#head = ['pose15x', 'pose15y', 'pose16x', 'pose16y', 'pose17x', 'pose17y', 'pose18x', 'pose18y']
gaze_angle = ['gaze_angle_x', 'gaze_angle_y']
