"""file for all the recurent neural network, will have to make a LSTM and a GRU model"""

import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils
from matplotlib import pyplot as plt
#from matplotlib.backend_bases import FigureCanvasBase
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, mean_squared_error
import scipy.special
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lrs
from net import LSTMModel3
from shutil import copyfile
import VennABERS
from sklearn.linear_model import LogisticRegression as LR
import fix_random_seed
import random
from captum.attr import IntegratedGradients
from tqdm import tqdm
from data_formatting import split_sequence_overlap, split_sequence_nooverlap, split_sequence, split_train_test, normalize_data, set_targets

import parameters

parameters.initialize_parameters()


if len(sys.argv)!=3:
    print("Usage: %s <input_dir> <explain>\n" % sys.argv[0])
    sys.exit(0)

input_dir = sys.argv[1]
explain = bool(int(sys.argv[2]))


regression = False
regresssion_classification_thresh = 0.5

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# force device
#device = 'cpu'
print(f"Using {device}")

fix_random_seed.fix_random_seed()

# Get data
#train = pd.read_csv("/data/private/eveilcoma/csv_4_NN/Concatenated_csv/Diff_Module_All.csv",  delimiter=";")
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules.csv",  delimiter=",") # Diff Modules
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Modules.csv",  delimiter=";")   # Modules
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Positions.csv",  delimiter=",")   # Positions
train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_new_withoutAUc.csv",  delimiter=",")  # 101 features (only AU_r) (filtered)
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL/All_Subs_Diff_Modules_not_norm.csv",  delimiter=",")  # 101 features (only AU_r) (filtered)
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_DL_nofilter/All_Subs_Diff_Modules_nofilter_withoutAUc.csv",  delimiter=",")  # 101 features (only AU_r) (not filtered)


train_df, nclasses, targets_numpy = set_targets(train_df)

# Convert the subject names (strings) into numbers
subjects = pd.factorize(train_df['Subject'])[0]


# extract + normalise the features from dataframe
features_numpy = normalize_data(train_df, parameters.normalise_individual_subjects)
input_dim = features_numpy.shape[1]
print(f"Number of features: {input_dim}")



test_accuracies = []
test_bal_accuracies = []
test_auc_scores = []
calibrated_test_accuracies = []
avg_all_attr = np.empty((0, input_dim), dtype='float')

subj = np.unique(subjects)

for test_subj in subj:
  print(f"Processing subject {test_subj}")
  xv_max_val = 0
  #avg_test_acc = 0
  #avg_bal_test_acc = 0
  val_acc_val_loss_list = []
  test_acc_list = []
  test_auc_list = []
  test_bal_acc_list = []
  avg_attr_persubject = np.zeros((input_dim), dtype='float')
  for xv in range(parameters.cross_validation_passes):
    #if test_subj<2:
      #continue

    test_idx = np.array([test_subj])
    trainval_idx = np.delete(subj, np.where(subj==test_subj)) # take out test subject from trainval
    #val_idx = [trainval_idx[-1:]] # use last subject in trainval set for validation
    #val_idx = trainval_idx[-2:] # use last subject in trainval set for validation
    val_idx = np.array([test_subj+1, test_subj+2, test_subj+3]) # use three following subjects for validation
    #val_idx = np.array([test_subj+1, test_subj+2, test_subj+3, test_subj+4]) # use four following subjects for validation
    #val_idx = trainval_idx[random.sample(range(len(trainval_idx)), num_validation_subjects)]
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

    print(f"Number of test examples: {len(targets_test)}")

    # create feature and targets tensor for test set.
    if parameters.test_with_subsequences:
      featuresTest = torch.from_numpy(features_test)
      if regression:
        targetsTest = torch.from_numpy(targets_test).type(torch.FloatTensor)  
      else:
        targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)  # data type is long
      test = TensorDataset(featuresTest, targetsTest)
      test_loader = DataLoader(test, batch_size=parameters.batch_size, shuffle=False)


    model = LSTMModel3(input_dim, parameters.hidden_dim, parameters.layer_dim, nclasses, device)


    if regression:
      error = nn.MSELoss().to(device)
      error_cpu = nn.MSELoss().to('cpu')
    else:
      # Cross Entropy Loss
      if nclasses==3:
        weights =  torch.tensor([0.3, 0.3, 0.4]).to(device)
        error = nn.CrossEntropyLoss(weight=weights).to(device)
        weights_cpu =  torch.tensor([0.3, 0.3, 0.4]).to('cpu')
        error_cpu = nn.CrossEntropyLoss(weight=weights).to('cpu')
      else:
        error = nn.CrossEntropyLoss()
        error_cpu = nn.CrossEntropyLoss().to('cpu')


    model_filename = f"{input_dir}/checkpoint_{test_subj:02}_{xv:02}.ckpt"

    # Test
    model.load_state_dict(torch.load(model_filename))
    correct = 0
    total = 0
    prev_label = -1
    class_hist = np.zeros(nclasses, dtype='int')
    all_predicted = []
    all_labels = []
    all_outputs = np.empty((0, nclasses), dtype='float')
    # Iterate through test dataset
    model.eval()
    with torch.no_grad():
      if parameters.test_with_subsequences:
        for features, labels in test_loader:
          features = Variable(features.view(-1, parameters.test_seq_dim, input_dim)).to(device)
          labels = Variable(labels).to('cpu')

          # Forward propagation
          outputs = model(features)

          if regression and outputs.ndim>1:
            test_loss = error_cpu(torch.squeeze(outputs).to('cpu'), labels)
          else:
            test_loss = error_cpu(outputs.to('cpu'), labels)
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
            correct += (predicted == labels).sum().item()
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
        
      else:  # not test_with_subsequences
        count=0
        for features in features_test:
          features = torch.tensor(features)
          features = torch.unsqueeze(features, 0).to(device)
          labels = torch.unsqueeze(torch.tensor(targets_test[count]), 0)
          #features = Variable(features.view(-1, seq_dim, input_dim)).to(device)

          # Forward propagation
          outputs = model(features)

          if regression and outputs.ndim>1:
            test_loss = error(torch.squeeze(outputs).to('cpu'), labels)
          else:
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

    al_np = np.array(all_labels)   
    ao_np = np.array(all_outputs)  
    ap_np = np.array(all_predicted)
    if regression:
      if ao_np.ndim>1:
        ao_np2 = np.squeeze(ao_np)
      else:
        ao_np2 = ao_np
      accuracy = mean_squared_error(al_np, ao_np2)
      target = (al_np>regresssion_classification_thresh)
      bal_accuracy = ((al_np>regresssion_classification_thresh) == (ao_np2>regresssion_classification_thresh)).sum()/al_np.size
      if not (target==0).any():
        target[0]=0
      if not (target==1).any():
        target[0]=1
      auc = roc_auc_score(target, ao_np2)
    else: # classification
      accuracy = correct / float(total)
      bal_accuracy = balanced_accuracy_score(al_np, ap_np)
      if not (al_np==0).any():
        al_np[0]=0
      if not (al_np==1).any():
        al_np[0]=1
      auc = roc_auc_score(al_np, scipy.special.softmax(all_outputs)[:,1])
    print(f"Test accuracy for subject {test_subj} run {xv}: {accuracy}    AUC: {auc}")
    #avg_test_acc += accuracy
    #avg_test_auc += auc
    #avg_bal_test_acc += bal_accuracy
    test_acc_list.append(accuracy)
    test_bal_acc_list.append(bal_accuracy)
    test_auc_list.append(auc)


    ### Compute attributions (XAI)
    if explain:
      print("Computing feature attributions...")
      model.train()
      ig = IntegratedGradients(model)

      features = featuresTest[targetsTest.detach().numpy()==ap_np].to(device)
      targets = targetsTest[targetsTest.detach().numpy()==ap_np].to(device)
      feature_dim = features.shape[2]

      all_attr = np.empty((0, feature_dim), dtype='float')
      for i in tqdm(range(0, targets.shape[0], parameters.batch_size)):
        end_index = min(i+parameters.batch_size, targets.shape[0])
        attr, delta = ig.attribute(features[i:end_index], target=targets[i:end_index], return_convergence_delta=True)
        attr_np = attr.detach().cpu().numpy()
        attr_np_mean = np.mean(attr_np, axis=1)  # mean over temporal window
        all_attr = np.vstack((all_attr, attr_np_mean)) # stack over all batches
      avg_attr_persubject += np.mean(all_attr, axis=0) # add attr of one crossval run
  avg_attr_persubject /= parameters.cross_validation_passes
  avg_all_attr = np.vstack((avg_all_attr, avg_attr_persubject))

  avg_test_acc = np.mean(test_acc_list)
  avg_bal_test_acc = np.mean(test_bal_acc_list)
  avg_test_auc = np.mean(test_auc_list)

  #if explain:
    #avg_all_attr /= (parameters.cross_validation_passes*len(subj))
  if explain:
      np.savetxt("feature_attributions.csv", avg_all_attr, delimiter=',')

  # avg_test_acc /= parameters.cross_validation_passes
  test_accuracies.append(avg_test_acc)
  test_bal_accuracies.append(avg_bal_test_acc)
  test_auc_scores.append(avg_test_auc)
  print("Test accuracies:")
  print(test_accuracies)
  print(f"Mean accuracy: {np.mean(test_accuracies)}")
  #print(f"Variance: {np.var(test_accuracies)}")
  print(f"Stdev: {np.std(test_accuracies)}")
  print(f"Std error: {np.std(test_accuracies)/np.sqrt(np.size(test_accuracies))}")
  print(f"Avg AUC: {np.mean(test_auc_scores)}")
  print(f"AUC stdev: {np.std(test_auc_scores)}")
  print(f"AUC std error: {np.std(test_auc_scores)/np.sqrt(np.size(test_auc_scores))}")
  print("\n")
  print("Balanced test accuracies:")
  print(test_bal_accuracies)
  print(f"Mean balanced accuracy: {np.mean(test_bal_accuracies)}\n")

