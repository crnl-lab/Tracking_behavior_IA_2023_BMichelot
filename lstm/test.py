import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from net import LSTMModel3
import temperature_scaling
from sklearn.linear_model import LogisticRegression as LR
from data_formatting import split_sequence_overlap, split_sequence_nooverlap, split_sequence, split_train_test, normalize_data, set_targets
import parameters

parameters.initialize_parameters()


if len(sys.argv)!=4:
    print("Usage: %s <model_file> <test_subject> <calibration>\n" % sys.argv[0])
    sys.exit(0)

model_filename = sys.argv[1]
test_subj = int(sys.argv[2])
calibration = bool(int(sys.argv[3]))

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# override
#device = 'cpu'

print(f"Using {device}")

# Get data
#train = pd.read_csv("/data/private/eveilcoma/csv_4_NN/Concatenated_csv/Diff_Module_All.csv",  delimiter=";")
train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv",  delimiter=",")  # 101 features (only AU_r)
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL_new/All_Subs_Diff_Modules_new.csv",  delimiter=",")  # 118 features (all AU)
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL_new/All_Subs_Diff_Modules_new.csv",  delimiter=",")  # 118 features (all AU)

# data_tensor = torch.tensor(train.to_numpy())
# print(data_tensor)

train_df, nclasses, targets_numpy = set_targets(train_df)

# Convert the subject names (strings) into numbers
subjects = pd.factorize(train_df['Subject'])[0]


# normalise the features
features_numpy = normalize_data(train_df, parameters.normalise_individual_subjects)
input_dim = features_numpy.shape[1]
print(f"Number of features: {input_dim}")


subj = np.unique(subjects)


# old code
#test_idx = [test_subj]
#trainval_idx = np.delete(subj, np.where(subj==test_subj)) # take out test subject from trainval
#val_idx = [trainval_idx[-2:]] # use last subject in trainval set for validation
#train_idx = trainval_idx[0:-2]
# end old code

test_idx = np.array([test_subj])
trainval_idx = np.delete(subj, np.where(subj==test_subj)) # take out test subject from trainval
#val_idx = [trainval_idx[-1:]] # use last subject in trainval set for validation
#val_idx = trainval_idx[-2:] # use last subject in trainval set for validation
#val_idx = np.array([test_subj+1, test_subj+2, test_subj+3]) # use three following subjects for validation
val_idx = np.array([test_subj+1, test_subj+2, test_subj+3, test_subj+4]) # use four following subjects for validation
val_idx = val_idx%len(subj)
#train_idx = trainval_idx[0:-1]
#train_idx = trainval_idx[0:-2]
train_idx = np.setxor1d(subj, test_idx)
train_idx = np.setxor1d(train_idx, val_idx)

print("Generating train/val/test split...")
features_train, targets_train, features_val, targets_val, features_test, targets_test = split_train_test(targets_numpy, features_numpy, subjects, train_idx, val_idx, test_idx)

print("Generating sequences...")
if parameters.test_with_subsequences:
  features_test, targets_test = split_sequence_overlap(features_test, targets_test, parameters.test_seq_dim, parameters.test_overlap_size)
else:
  features_test, targets_test = split_sequence_nooverlap(features_test, targets_test, parameters.test_seq_dim, parameters.test_overlap_size)

#print(f"Number of training examples: {len(targets_train)}")
#print(f"Number of validation examples: {len(targets_val)}")
print(f"Number of test examples: {len(targets_test)}")


# create feature and targets tensor for test set.
if parameters.test_with_subsequences:
    featuresTest = torch.from_numpy(features_test)
    targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)  # data type is long
    test = TensorDataset(featuresTest, targetsTest)
    test_loader = DataLoader(test, batch_size=parameters.batch_size, shuffle=False)

# validation
features_val, targets_val = split_sequence_overlap(features_val, targets_val, parameters.seq_dim, parameters.overlap_size)
featuresVal = torch.from_numpy(features_val)
targetsVal = torch.from_numpy(targets_val).type(torch.LongTensor)  # data type is long
val = TensorDataset(featuresVal, targetsVal)
val_loader = DataLoader(val, batch_size=parameters.batch_size, shuffle=False)


print("Loading model.")
model = LSTMModel3(input_dim, parameters.hidden_dim, parameters.layer_dim, nclasses, device)

#learning_rate = 0.0005 # SGD
#learning_rate = 0.0005 # Adam # BEST
#learning_rate = 0.00001 # Adam # BEST
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10e-4)


# Cross Entropy Loss
error = nn.CrossEntropyLoss()


loss_list = []
val_loss_list = []
iteration_list = []
accuracy_list = []
val_accuracy_list = []
count = 0
min_val_loss = 1e6
max_val_accuracy = 0
clip_value=0.5


# Test
model.load_state_dict(torch.load(model_filename, map_location=device))

#scaled_model = temperature_scaling.ModelWithTemperature(model)
#scaled_model.set_temperature(val_loader)

if calibration:
  #test_model = scaled_model
  print("Using calibrated model.")
#else:
#  test_model = model
test_model = model


correct = 0
total = 0
prev_label = -1
class_hist = np.zeros(nclasses, dtype='int')
all_predicted = []
all_labels = []
all_outputs = np.empty((0, nclasses), dtype='float')

# Iterate through test dataset
test_model.eval()
with torch.no_grad():
  if parameters.test_with_subsequences:
    for features, labels in test_loader:
      features = Variable(features.view(-1, parameters.test_seq_dim, input_dim)).to(device)

      # Forward propagation
      outputs = test_model(features)

      test_loss = error(outputs.to('cpu'), labels)
      # Get predictions from the maximum value
      predicted = torch.max(outputs.data, 1)[1]
      predicted = predicted.to('cpu')
      if parameters.test_use_max:
        bi = 0
        for l in labels:
          if l!=prev_label and prev_label!=-1:
            final_predicted = np.argmax(class_hist)
            #print(class_hist)
            #print(final_predicted)
            #print(prev_label)
            if final_predicted==prev_label:
              correct += 1
            class_hist = np.zeros(nclasses, dtype='int')
            total += 1
            all_predicted.append(final_predicted)
            all_labels.append(l)

          class_hist[predicted[bi]] += 1
          prev_label = l
          bi += 1
      else:
        #print(predicted.device)
        #print(labels.device)

        # Total number of labels
        total += labels.size(0)
        correct += (predicted == labels).sum()
        all_predicted.extend(list(predicted.detach().numpy()))
        all_labels.extend(list(labels.detach().numpy()))
        all_outputs = np.concatenate((all_outputs, outputs.data.to('cpu').reshape(-1, nclasses)))

    if parameters.test_use_max and np.sum(class_hist)>0:
      final_predicted = np.argmax(class_hist)
      if final_predicted==prev_label:
        correct += 1
      total += 1
      all_predicted.append(final_predicted)
      all_labels.append(l)
    
  else:
    count=0
    for features in features_test:
      features = torch.tensor(features)
      features = torch.unsqueeze(features, 0).to(device)
      labels = torch.unsqueeze(torch.tensor(targets_test[count]), 0)
      #features = Variable(features.view(-1, seq_dim, input_dim)).to(device)

      # Forward propagation
      outputs = test_model(features)

      test_loss = error(outputs.to('cpu'), labels)
      # Get predictions from the maximum value
      predicted = torch.max(outputs.data, 1)[1]
      predicted = predicted.to('cpu')
      #print(predicted.device)
      #print(labels.device)

      # Total number of labels
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      count += 1

accuracy = correct / float(total)
print(f"Test accuracy: {accuracy}")


# Platt scaling
# Calculate validation accuracy
if calibration:
  all_val_predicted = []
  all_val_labels = []
  all_val_outputs = np.empty((0, nclasses), dtype='float')

  correct = 0
  total = 0
  # Iterate through validation dataset
  model.eval()
  with torch.no_grad():
    for features, labels in val_loader:
      features = Variable(features.view(-1, parameters.seq_dim, input_dim)).to(device)
      labels = Variable(labels).to(device)

      # Forward propagation
      outputs = model(features)

      val_loss = error(outputs, labels)
      # Get predictions from the maximum value
      predicted = torch.max(outputs.data, 1)[1]
      predicted = predicted.to('cpu')
      #print(predicted.device)
      #print(labels.device)

      # Total number of labels
      total += labels.size(0)
      correct += (predicted == labels.cpu()).sum()
      all_val_predicted.extend(list(predicted.detach().numpy()))
      all_val_labels.extend(list(labels.cpu().detach().numpy()))
      all_val_outputs = np.concatenate((all_val_outputs, outputs.data.to('cpu').reshape(-1, nclasses)))

  val_accuracy = correct / float(total)

  lr = LR()
  avl_np = np.array(all_val_labels) # val
  lr.fit(all_val_outputs, avl_np)
  pred = lr.predict_proba(all_outputs)
  mpred = np.argmax(pred, axis=1) 
  calibrated_accuracy = (all_labels==mpred).sum()/len(all_labels)

  print(f"Calibrated test accuracy: {calibrated_accuracy}")

