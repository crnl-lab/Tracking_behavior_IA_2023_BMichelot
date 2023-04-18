import numpy as np
import pandas as pd
from sklearn import preprocessing

# split between train and test, all first event goes into test the rest in train
def split_train_test(targets_numpy, features_numpy, subjects, train_subjects, val_subjects, test_subjects):
    train_idx = np.bitwise_or.reduce(np.array([subjects==i for i in train_subjects]), axis=0)
    val_idx = np.bitwise_or.reduce(np.array([subjects==i for i in val_subjects]), axis=0)
    test_idx = np.bitwise_or.reduce(np.array([subjects==i for i in test_subjects]), axis=0)

    features_train = features_numpy[train_idx]
    targets_train = targets_numpy[train_idx]
    features_val = features_numpy[val_idx]
    targets_val = targets_numpy[val_idx]
    features_test = features_numpy[test_idx]
    targets_test = targets_numpy[test_idx]

    return features_train, targets_train, features_val, targets_val, features_test, targets_test



def split_sequence(features, labels, window_size):
    """sépare les données (en format frame par frame) dans un tableau en 3d par séquence de window_size frame

    window_size : frame"""

    ret_features, ret_labels, seq_features, seq_labels = [], [], [], []
    framecount = 0
    last_event = labels[0]
    for i in range(len(features)):
        seq_features.append(features[i])
        # seq_labels.append(labels[i])

        if framecount == window_size - 1:
            ret_features.append(np.array(seq_features))
            if (labels[i] == 65) or (labels[i] == 66):
                ret_labels.append(0)
            if (labels[i] == 67) or (labels[i] == 68):
                ret_labels.append(1)
            if (labels[i] == 69) or (labels[i] == 70):
                ret_labels.append(2)
            seq_features = []
            seq_labels = []
            framecount = -1

        if labels[i] != last_event:
            last_event = labels[i]
            seq_features = []
            seq_labels = []
            framecount = -1
        framecount += 1

    ret_features = np.array(ret_features)
    ret_labels = np.array(ret_labels)
    return ret_features, ret_labels


def split_sequence_overlap(features, labels, window_size, step_size):
    ret_features, ret_labels = [], []
    for i in range(0, len(features)-window_size+1, step_size):
        if labels[i]==labels[i+window_size-1]:
            ret_features.append(features[i:i+window_size])
            ret_labels.append(labels[i])
    return np.array(ret_features), np.array(ret_labels)

def split_sequence_nooverlap(features, labels, window_size, step_size):
    ret_features, ret_labels = [], []
    start_idx=0
    for i in range(1, len(features)):
        if labels[i]!=labels[i-1]:
          ret_features.append(features[start_idx:i])
          ret_labels.append(labels[i-1])
          start_idx=i
    ret_features.append(features[start_idx:len(features)])
    ret_labels.append(labels[len(features)-1])
    return np.array(ret_features, dtype=object), np.array(ret_labels)


def normalize_data(train_df, normalise_individual_subjects):
  #train_df = train.drop(train.columns[0], axis=1)  # drop index column
  #train_df = train.drop(['Condition','Subject','index'], axis= 1)
  subjects = pd.factorize(train_df['Subject'])[0]
  train_df = train_df.drop(['Condition','Subject','Emotion','Presence'], axis= 1)
  features_numpy = train_df.to_numpy(dtype='float32')

  unique_values = np.array([len(np.unique(features_numpy[:,i])) for i in range(features_numpy.shape[1])])
  floatcols = unique_values!=2  # retrieve all columns that are not boolean
  if normalise_individual_subjects:
    for i in np.unique(subjects): # normalise each subject individually
      scaler = preprocessing.StandardScaler()  # BEST
      #scaler = preprocessing.MinMaxScaler()
      (features_numpy[(subjects==i)])[:, floatcols] = scaler.fit_transform((features_numpy[(subjects==i)])[:, floatcols])
  else:
    scaler = preprocessing.StandardScaler()  # BEST
    features_numpy[:, floatcols] = scaler.fit_transform(features_numpy[:, floatcols])
  return features_numpy


def set_targets(train_df):
  # Drop "repos" conditions 
  #train_df = train_df.drop(train_df[train_df['Condition'] == 65].index) # stimulation
  #train_df = train_df.drop(train_df[train_df['Condition'] == 66].index) # stimulation
  train_df = train_df.drop(train_df[train_df['Condition'] == 67].index) # repos
  train_df = train_df.drop(train_df[train_df['Condition'] == 68].index) # repos
  train_df = train_df.drop(train_df[train_df['Condition'] == 69].index) # interaction
  train_df = train_df.drop(train_df[train_df['Condition'] == 70].index) # interaction
  #train_df = train_df.drop(train_df[(train_df['Emotion'] > 0.2) & (train_df['Emotion'] < 0.8)].index)
  #train_df = train_df.drop(train_df[(train_df['Presence'] > 0.2) & (train_df['Presence'] < 0.8)].index)

  targets_numpy = train_df.Condition.values

  nclasses = 2
  # 1) Classification des conditions
  # Replace the condition codes (65..70) with target values (0..n)
  targets_numpy[targets_numpy==65] = 0  # Stimulation : neutre
  targets_numpy[targets_numpy==66] = 1  # Stimulation : emotionnelle
  #targets_numpy[targets_numpy==67] = 0  # Repos : neutre
  #targets_numpy[targets_numpy==68] = 0  # Repos : emotionnel
  #targets_numpy[targets_numpy==69] = 0  # Interaction : neutre
  #targets_numpy[targets_numpy==70] = 1  # Interaction : emotionnelle

  '''
  nclasses = 2
  # 2) Estimations subjectives des sujets :
  # Emotion
  targets_numpy[train_df.Emotion.values<0.6] = 0
  targets_numpy[train_df.Emotion.values>=0.6] = 1
  # Présence
  #targets_numpy[train_df.Presence.values<0.5] = 0
  #targets_numpy[train_df.Presence.values>=0.5] = 1
  '''

  '''
  # regression
  nclasses = 1
  #targets_numpy = np.log(1+train_df.Emotion.values)
  #targets_numpy = np.power(train_df.Emotion.values,2)
  targets_numpy = train_df.Emotion.values
  #targets_numpy = train_df.Presence.values
  
  # normalise targets
  subjects = pd.factorize(train_df['Subject'])[0]
  for i in np.unique(subjects): # normalise each subject individually
    scaler = preprocessing.StandardScaler()  # BEST
    #scaler = preprocessing.MinMaxScaler()
    targets_numpy[(subjects==i)] = np.squeeze(scaler.fit_transform(np.expand_dims(targets_numpy[(subjects==i)],1)))
    '''

  return train_df, nclasses, targets_numpy


