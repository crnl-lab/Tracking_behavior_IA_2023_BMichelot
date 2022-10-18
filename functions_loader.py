import os
import csv
import json
import struct
import warnings
import cv2


import config as cfg
import filter as flt
import convert_json as cvt


############################sub functions needed for loader##################

def read_json(json_file):
    """read json file"""

    with open(json_file, 'r') as f:
        return json.load(f)

def read_csv(csv_file):
    """read csv file"""

    ret = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            ret.append(row)

    return ret

def write_csv(path_csv, data):
    """write the data (2D matrix) in the given location
    creates the file if it doesn't exist yet"""
    with open(path_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def write_json(path_json,data):
    """"""
    with open(path_json, 'w', newline='') as f:
        json.dump(data, f)

def set_face(path_face_csvfile):
    """returns a 2d matrix containing all the OpenFace data exept the 3d points"""
    face = read_csv(path_face_csvfile)
    for i in range(len(face)):
        for y in range(638, 434, -1):
            face[i].pop(y)
        for y in range(292, 124, -1):
            face[i].pop(y)

    return face

def set_pose(path_pose_folder):
    """read all the json file from the OpenPose_path in its basic format"""
    pose = []
    # parcours tout les fichiers json du dossier
    for filename in os.listdir(path_pose_folder):
        if filename.endswith(".json"):
            dict = read_json(path_pose_folder + filename)
            pose.append(dict)
    print("number of frame = " + str(len(pose)))
    return pose


def reformat_pose(pose, faulty_frames):
    """Transforme the weird dictionnary format of OpenPose into 2d matrix [frames][features]"""
    ret = []
    header = []
    header_size = -1
    ret_frame = []
    frame_num = 0
    for frame in pose:

        if (len(frame['people']) > 1):
            #TODO BRUNO DECOMMENTER
            #print("Warning : plusieurs personnes sont détéctés par OpenPose à la frame " + str(frame_num))
            faulty_frames.append(frame_num)


        if (len(frame['people']) == 0):
            # TODO BRUNO DECOMMENTER
            #print("Warning : il n'y personne de détécté sur l'image à la frame " + str(frame_num))
            faulty_frames.append(frame_num)
            frame_num += 1
            ret_frame = [0] * header_size
            ret.append(ret_frame)
            ret_frame = []
            continue

        # add all the point of the body
        feat_count = 0
        for pose in frame['people'][0]['pose_keypoints_2d']:
            # add the feature name to the header in the first loop
            if frame_num == 0 and feat_count < 19 * 3:
                if feat_count % 3 == 0:
                    header.append('pose' + str(feat_count // 3) + 'x')
                if feat_count % 3 == 1:
                    header.append('pose' + str(feat_count // 3) + 'y')
                if feat_count % 3 == 2:
                    header.append('pose' + str(feat_count // 3) + 'conf')
            feat_count += 1

            # add the feature value to the frame list at every loop
            if feat_count < 19 * 3 + 1:
                ret_frame.append(float(pose))

        # face
        feat_count = 0
        for face in frame['people'][0]['face_keypoints_2d']:
            if frame_num == 0:
                if feat_count % 3 == 0:
                    header.append('face' + str(feat_count // 3) + 'x')
                if feat_count % 3 == 1:
                    header.append('face' + str(feat_count // 3) + 'y')
                if feat_count % 3 == 2:
                    header.append('face' + str(feat_count // 3) + 'conf')
            feat_count += 1

            ret_frame.append(float(face))

        # left hand
        feat_count = 0
        for hand_l in frame['people'][0]['hand_left_keypoints_2d']:
            if frame_num == 0:
                if feat_count % 3 == 0:
                    header.append('lefthand' + str(feat_count // 3) + 'x')
                if feat_count % 3 == 1:
                    header.append('lefthand' + str(feat_count // 3) + 'y')
                if feat_count % 3 == 2:
                    header.append('lefthand' + str(feat_count // 3) + 'conf')
            feat_count += 1

            ret_frame.append(float(hand_l))

        # right_hand
        feat_count = 0
        for hand_r in frame['people'][0]['hand_right_keypoints_2d']:
            if frame_num == 0:
                if feat_count % 3 == 0:
                    header.append('righthand' + str(feat_count // 3) + 'x')
                if feat_count % 3 == 1:
                    header.append('righthand' + str(feat_count // 3) + 'y')
                if feat_count % 3 == 2:
                    header.append('righthand' + str(feat_count // 3) + 'conf')
            feat_count += 1

            ret_frame.append(float(hand_r))

        # add the header to ret at the first loop
        if frame_num == 0:
            header_size = len(header)
            ret.append(header)
        # add the frame list for every loop
        ret.append(ret_frame)
        ret_frame = []

        frame_num += 1

    return ret, faulty_frames

def set_tps_timeStamps(path_tps_folder):
        """Same function as get_tps_Timestamp_old differs only because it takes TPS file in milliseconds contrary from the first wich are floats in seconds"""

        tps_file = ""
        stamps_file = ""
        slideres_file = ""
        # selects the right tps and syncIN file from the folder
        # BE CAREFUL with the naming of the videos, this function works if the video of the patient is the third and the file finished with 2.tps
        #same as above
        for filename in os.listdir(path_tps_folder):
            if filename.endswith("2.tps"):
                tps_file = path_tps_folder + filename
            if filename.endswith(".syncIN"):
                stamps_file = path_tps_folder + filename
            if filename.endswith(".txt"):
                slideres_file = path_tps_folder + filename

        blocksize = 4

        Frame_time = []
        with open(tps_file, "rb") as f:
            while True:
                buf = f.read(blocksize)
                if not buf:
                    break
                value = struct.unpack('<L', buf)
                Frame_time.append(value[0])

        TimeStamps = []
        TimeStamps_file = open(stamps_file, 'r')
        lines = TimeStamps_file.readlines()
        for line in lines:
            temp = line.split("\t")
            if temp[0] == 'A' or temp[0] == 'B' or temp[0] == 'C' or temp[0] == 'D' or temp[0] == 'E' or temp[0] == 'F':
                time = float(temp[1][:11].replace(',', '.'))
                frame = 0
                start = 0
                while frame < len(Frame_time):
                    if (Frame_time[frame] > time):
                        start = frame
                        break
                    frame += 1

                while frame < len(Frame_time):
                    if (Frame_time[frame] > (time) + 60000):
                        break
                    frame += 1

                TimeStamps.append({'code': temp[0], 'time': time, 'start': start, 'end': frame - 1})

        slideres = open(slideres_file, 'r')
        lines = slideres.readlines()
        count = 0
        # rajoute la retour des témoins sur les échelles d'émotion et de présence dans le TimeStamps en les positionant entre -1 et 1
        for line in lines:
            temp = line.split("\t")
            if count % 2 == 0:
                TimeStamps[count // 2]['slideEmotion'] = float(temp[1]) / 450

            if count % 2 == 1:
                TimeStamps[count // 2]['slidePresence'] = float(temp[1]) / 450

            count += 1

        return Frame_time, TimeStamps


def regroup_data(pose, face, timestamps, times, faulty_frames):
        """regroups the data from pose face and times in a list. For every event their is the 2d matrix for the data and a dictionnary containing
        the number of the event, its code and the Emotional et Presence value given by the subject of the Event


        return : [nbr-event][[frames][features], {num : int ,code : str, slideEmotion : float , slidePresence : float}]"""

        # write the header with the times of the frame, pose header, face header then event
        header = ['time']
        for i in pose[0]:
            header.append(i)
        for i in face[0]:
            header.append(i)

        if (len(pose) != len(face)):
            warnings.warn('WARNING: pas le même nombre de frame entre OpenPose et OpenFace')
            print("nombre de frame openFace = " + str(len(face)))
            print("nombre de frame openPose = " + str(len(pose)))

        ret = []

        frame_data = []
        evmtcount = 0
        for evmt in timestamps:

            event_signature = {'num': evmtcount, 'code': evmt['code'], 'emotion': evmt['slideEmotion'],
                               'presence': evmt['slidePresence']}
            data = [header]
            start_time = times[evmt['start']]
            for framenum in range(evmt['start'], evmt['end']):
                if framenum in faulty_frames:
                    warnings.warn("ATTENTION il y a des frames avec soit trop de personnes soit zéro personnes pendant un évènement sur la frame : " + str(framenum))


                frame_data.append(times[framenum] - start_time)
                for i in pose[framenum]:
                    frame_data.append(i)
                for i in face[framenum]:
                    frame_data.append(i)
                data.append(frame_data)
                frame_data = []

            ret.append([data, event_signature])
            evmtcount += 1

        return ret


##############################################################################



def loader(path_face_csvfile, path_pose_folder, path_tps_folder, output_path, filter= True, write=True):
    
    # create list to remember all faulty frames (detects for OpenPose only)
    faulty_frames = []

    # get all data from different sources
    print('set_face')
    face = set_face(path_face_csvfile)
    print('set_pose')
    frame_array = set_pose(path_pose_folder)

    # filter the data
    if filter:
        pose_data, face_data, hand_left_data, hand_right_data = cvt.convert_json(None, json_data=frame_array, save_output=False)
        hand_left_data = flt.flat_median_filter(None, window_size=5, threshold=5.0, flat_window=10, flat_tolerance=0.1, save_output=False, raw=hand_left_data)
        hand_right_data = flt.flat_median_filter(None, window_size=5, threshold=5.0, flat_window=10, flat_tolerance=0.1, save_output=False, raw=hand_right_data)
        pose_data = flt.flat_median_filter(None, window_size=5, threshold=5.0, flat_window=10, flat_tolerance=0.1, save_output=False, raw=pose_data)
        face_data = flt.savgol(None, window_size=25, polynome_degree=2, save_output=False, time_serie=face_data)
        np_data = {
            "pose": pose_data,
            "face": face_data,
            "hand_left": hand_left_data,
            "hand_right": hand_right_data,
        }
        frame_array = cvt.convert_np(None, np_data= np_data, save_output= False)

    print('reformat')
    pose, faulty_frames = reformat_pose(frame_array, faulty_frames)

    print('change time stamp')
    ###change from old version if loading new data
    times, timestamps = set_tps_timeStamps(path_tps_folder)

    # regroup all data
    data = regroup_data(pose, face, timestamps, times, faulty_frames)

    subject_name = output_path[len(output_path) - 7: len(output_path) - 1]
    
    
    if write:
        # Now, writes the loaded data in the output path
        #for every event their is a data.csv file (2d matrix) and a signature.json file (dictionary)"""
        evmt_count = 0
        for evmt in data:
            print("en train d'écrire le fichier correspondant à l'évènement : " + str(evmt[1]))
            write_csv(output_path + subject_name + '_' + str(evmt_count) + '_' + str(evmt[1]['code']) + '_data.csv', evmt[0])
            write_json(output_path + subject_name + '_' + str(evmt_count) + '_' + str(evmt[1]['code']) + "_signature.json", evmt[1])
            evmt_count += 1
    
    
    return data, timestamps, subject_name
    

def cut_video(video_path, timestamps, output_path, subject_name):
    """ cut the video from the video path into smaller videos for every event in the output path"""

    #open video file
    vid = cv2.VideoCapture(video_path)

    # define width and height from input
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    #frame_rate changed by hand be careful if change in experiment values
    frame_rate = 20

    frame_num = 1
    evmt_count = 0
    for evmt in timestamps:
        print(evmt)
        start_frame = evmt['start']
        end_frame = evmt['end']

        while (frame_num < start_frame):
            vid.read()
            frame_num += 1

        # define the output writer
        out = cv2.VideoWriter(output_path  + subject_name + '_' + str(evmt_count) + '_' + str(evmt['code']) + '.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))

        while (frame_num < end_frame):
            # write video
            ret, frame = vid.read()
            if ret == True:
                out.write(frame)
                
                # affichage des vidéos en temps réelle, non nécessaire
                #cv2.imshow('frame', frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #   break

                frame_num += 1


        out.release()
        evmt_count += 1

    vid.release()
    cv2.destroyAllWindows()

    return None