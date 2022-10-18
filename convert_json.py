"""
JSON to Numpy converter for OpenPose outputs.
Author: Alexandre Bremard | INSA-Lyon
GitHub: TODO
"""

import json
import sys
import numpy as np

def convert_json(path, json_data=None, save_output=True):
    """
        This script will read the json output given by an OpenPose estimator and will parse the data into 4 separate files (face, pose, hand_left, hand_right)
        The output .npy file will contain a [25][3][-1] numpy array with 25 keypoints and 3 coordinates (x, y, certainty) for each keypoints.
        The last dimension is the number of OpenPose timeframes. The object will look like this:
            array([[[373.021   , 372.494   , 372.057   , ..., 373.063   ,
                    373.031   , 373.601   ],
                    [297.446   , 297.45    , 298.273   , ..., 294.755   ,
                    294.758   , 292.542   ],
                    [  0.655306,   0.691308,   0.711365, ...,   0.688218,
                    0.692544,   0.673469]],
        Since the keypoints are grouped together as time series, it will be easier to run a filter on each time serie.

        The output files will be named as the input + an extension. For example the body position data will be in the "{input}_pose.npy" file
    """
    if json_data is None:
        with open(path) as f:
            json_data = json.load(f)

    pose_keypoints_2d_x, pose_keypoints_2d_y, pose_keypoints_2d_conf = ([] for i in range(3))
    face_keypoints_2d_x, face_keypoints_2d_y, face_keypoints_2d_conf = ([] for i in range(3))
    hand_left_keypoints_2d_x, hand_left_keypoints_2d_y, hand_left_keypoints_2d_conf = ([] for i in range(3))
    hand_right_keypoints_2d_x, hand_right_keypoints_2d_y, hand_right_keypoints_2d_conf = ([] for i in range(3))

    for sample in json_data:

        if len(sample['people']) > 0:

            pose_keypoints_2d = sample['people'][0]['pose_keypoints_2d'] 
            face_keypoints_2d = sample['people'][0]['face_keypoints_2d']
            hand_left_keypoints_2d = sample['people'][0]['hand_left_keypoints_2d']
            hand_right_keypoints_2d = sample['people'][0]['hand_right_keypoints_2d']

            pose_keypoints_2d_x += pose_keypoints_2d[0::3]
            pose_keypoints_2d_y += pose_keypoints_2d[1::3]
            pose_keypoints_2d_conf += pose_keypoints_2d[2::3]

            face_keypoints_2d_x += face_keypoints_2d[0::3]
            face_keypoints_2d_y += face_keypoints_2d[1::3]
            face_keypoints_2d_conf += face_keypoints_2d[2::3]

            hand_left_keypoints_2d_x += hand_left_keypoints_2d[0::3]
            hand_left_keypoints_2d_y += hand_left_keypoints_2d[1::3]
            hand_left_keypoints_2d_conf += hand_left_keypoints_2d[2::3]

            hand_right_keypoints_2d_x += hand_right_keypoints_2d[0::3]
            hand_right_keypoints_2d_y += hand_right_keypoints_2d[1::3]
            hand_right_keypoints_2d_conf += hand_right_keypoints_2d[2::3]

        else :

            pose_keypoints_2d_x += [0 for i in range(25)]
            pose_keypoints_2d_y += [0 for i in range(25)]
            pose_keypoints_2d_conf += [0 for i in range(25)]

            face_keypoints_2d_x += [0 for i in range(70)]
            face_keypoints_2d_y += [0 for i in range(70)]
            face_keypoints_2d_conf += [0 for i in range(70)]

            hand_left_keypoints_2d_x += [0 for i in range(21)]
            hand_left_keypoints_2d_y += [0 for i in range(21)]
            hand_left_keypoints_2d_conf += [0 for i in range(21)]

            hand_right_keypoints_2d_x += [0 for i in range(21)]
            hand_right_keypoints_2d_y += [0 for i in range(21)]
            hand_right_keypoints_2d_conf += [0 for i in range(21)]


    pose_data = np.zeros((25, 3, int(len(pose_keypoints_2d_x)/25)))
    face_data = np.zeros((70, 3, int(len(face_keypoints_2d_x)/70)))
    hand_left_data = np.zeros((21, 3, int(len(hand_left_keypoints_2d_x)/21)))
    hand_right_data = np.zeros((21, 3, int(len(hand_right_keypoints_2d_x)/21)))


    for i in range(25):
        pose_data[i][0] = pose_keypoints_2d_x[i::25]
        pose_data[i][1] = pose_keypoints_2d_y[i::25]
        pose_data[i][2] = pose_keypoints_2d_conf[i::25]

    for i in range(70):
        face_data[i][0] = face_keypoints_2d_x[i::70]
        face_data[i][1] = face_keypoints_2d_y[i::70]
        face_data[i][2] = face_keypoints_2d_conf[i::70]
        
    for i in range(21):
        hand_left_data[i][0] = hand_left_keypoints_2d_x[i::21]
        hand_left_data[i][1] = hand_left_keypoints_2d_y[i::21]
        hand_left_data[i][2] = hand_left_keypoints_2d_conf[i::21]
        
        hand_right_data[i][0] = hand_right_keypoints_2d_x[i::21]
        hand_right_data[i][1] = hand_right_keypoints_2d_y[i::21]
        hand_right_data[i][2] = hand_right_keypoints_2d_conf[i::21]         

    if save_output:

        name_wo_extension = path.split(".")[0]
        np.save(f"{name_wo_extension}_pose.npy", pose_data)
        np.save(f"{name_wo_extension}_face.npy", face_data)
        np.save(f"{name_wo_extension}_hand_left.npy", hand_left_data)
        np.save(f"{name_wo_extension}_hand_right.npy", hand_right_data)

    return pose_data, face_data, hand_left_data, hand_right_data

def convert_np(np_path, np_data= None, save_output= True):
    """ Reverse function of convert_json()
    """
    
    if np_data is None:
        pose_data = np.load(f"{np_path}_pose_filtered.npy")
        face_data = np.load(f"{np_path}_face_filtered.npy")
        hand_left_data = np.load(f"{np_path}_hand_left_filtered.npy")
        hand_right_data = np.load(f"{np_path}_hand_right_filtered.npy")
        output_path = f"{np_path}_filtered.json"
    else:
        pose_data = np_data["pose"]
        face_data = np_data["face"]
        hand_left_data = np_data["hand_left"]
        hand_right_data = np_data["hand_right"]

    nb_frame = max([len(pose_data[0][0]), len(face_data[0][0]), len(hand_left_data[0][0]), len(hand_right_data[0][0])])
    final_dict =  []

    for frame in range(nb_frame):
        frame_dict = {
            "version": 1.3,
            "people": [{ "person_id": [-1] }]
        }
        pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d = ([] for i in range(4))

        for i in range(25):
            pose_keypoints_2d.append(pose_data[i][0][frame])
            pose_keypoints_2d.append(pose_data[i][1][frame])
            pose_keypoints_2d.append(pose_data[i][2][frame])

        for i in range(70):
            face_keypoints_2d.append(face_data[i][0][frame])
            face_keypoints_2d.append(face_data[i][1][frame])
            face_keypoints_2d.append(face_data[i][2][frame])

        for i in range(21):
            hand_left_keypoints_2d.append(hand_left_data[i][0][frame])
            hand_left_keypoints_2d.append(hand_left_data[i][1][frame])
            hand_left_keypoints_2d.append(hand_left_data[i][2][frame])

            hand_right_keypoints_2d.append(hand_right_data[i][0][frame])
            hand_right_keypoints_2d.append(hand_right_data[i][1][frame])
            hand_right_keypoints_2d.append(hand_right_data[i][2][frame])                                    

        frame_dict["people"][0]["pose_keypoints_2d"] = pose_keypoints_2d
        frame_dict["people"][0]["face_keypoints_2d"] = face_keypoints_2d        
        frame_dict["people"][0]["hand_left_keypoints_2d"] = hand_left_keypoints_2d        
        frame_dict["people"][0]["hand_right_keypoints_2d"] = hand_right_keypoints_2d

        final_dict.append(frame_dict)

    if save_output:
        with open(output_path, 'w') as fp:
            json.dump(final_dict, fp)
    
    return final_dict

if __name__ == '__main__':

    path = sys.argv[1]
    # convert_json(path)
    convert_np(path)    
