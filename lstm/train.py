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
from sklearn.metrics import mean_squared_error
import scipy.special
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lrs
from net import LSTMModel3
from shutil import copyfile
#import VennABERS
from sklearn.linear_model import LogisticRegression as LR
import fix_random_seed
import random
from data_formatting import split_sequence_overlap, split_sequence_nooverlap, split_sequence, split_train_test, normalize_data, set_targets
import parameters
from torchsampler import ImbalancedDatasetSampler

parameters.initialize_parameters()


if len(sys.argv)!=2:
    print("Usage: %s <csvfile>\n" % sys.argv[0])
    sys.exit(0)

csvfile = sys.argv[1]

modeldir = "."
model_filename = "checkpoint.ckpt"
min_num_epochs = 50 # 80 # 100
min_epoch_of_max_val = 4  # min epoch of validation maximum
patience = 40 # 50
num_validation_subjects = 1 # number of subjects used for validation
#learning_rate = 0.0005 # SGD
#learning_rate = 0.0005 # Adam # BEST ?
#learning_rate = 0.0007 # Adam 
learning_rate = 0.0007 # Adam  # BEST 
#learning_rate = 0.0005 # SGD  # BEST
momentum = 0.9
beta1 = 0.99
beta2 = 0.999
weight_decay = 10e-4
clip_value=0.1
data_augmentation = False
gaussian_noise_std = 0.01
reinit = True
do_plots = False
remove_lowest_run = False
regression = False
balance_classes = False

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

fix_random_seed.fix_random_seed()

# Get data
#train = pd.read_csv("/data/private/eveilcoma/csv_4_NN/Concatenated_csv/Diff_Module_All.csv",  delimiter=";")
train_df = pd.read_csv(csvfile,  delimiter=",")  # 101 features (only AU_r)
#train_df = pd.read_csv("/data/private/eveilcoma/temoins2022/Tables_for_DL_new/All_Subs_Diff_Modules_new.csv",  delimiter=",")  # 118 features (all AU)

train_df, nclasses, targets_numpy = set_targets(train_df)

# Convert the subject names (strings) into numbers
subjects = pd.factorize(train_df['Subject'])[0]

# normalise the features
features_numpy = normalize_data(train_df, parameters.normalise_individual_subjects)
input_dim = features_numpy.shape[1]
print(f"Number of features: {input_dim}")

del train_df

if do_plots:
  plt.ioff()
  #plt.figure()
  plt.clf()
  plt.subplot(441)

test_accuracies = []
calibrated_test_accuracies = []


subj = np.unique(subjects)

for test_subj in subj:
  xv_max_val = 0
  avg_test_acc = 0
  val_acc_val_loss_list = []
  test_acc_list = []
  for xv in range(parameters.cross_validation_passes):
    #if test_subj<2:
      #continue

    test_idx = np.array([test_subj])
    trainval_idx = np.delete(subj, np.where(subj==test_subj)) # take out test subject from trainval
    #val_idx = [trainval_idx[-1:]] # use last subject in trainval set for validation
    #val_idx = trainval_idx[-2:] # use last subject in trainval set for validation
    #val_idx = np.array([test_subj+1, test_subj+2, test_subj+3]) # use three following subjects for validation
    #val_idx = np.array([test_subj+1, test_subj+2, test_subj+3, test_subj+4]) # use four following subjects for validation
    val_idx = trainval_idx[random.sample(range(len(trainval_idx)), num_validation_subjects)]
    val_idx = val_idx%len(subj)
    #train_idx = trainval_idx[0:-1]
    #train_idx = trainval_idx[0:-2]
    train_idx = np.setxor1d(subj, test_idx)
    train_idx = np.setxor1d(train_idx, val_idx)

    print("Generating train/val/test split...")
    features_train, targets_train, features_val, targets_val, features_test, targets_test = split_train_test(targets_numpy, features_numpy, subjects, train_idx, val_idx, test_idx)

    print("Generating sequences...")
    features_train, targets_train = split_sequence_overlap(features_train, targets_train, parameters.seq_dim, parameters.overlap_size)
    features_val, targets_val = split_sequence_overlap(features_val, targets_val, parameters.seq_dim, parameters.overlap_size)
    if parameters.test_with_subsequences:
      features_test, targets_test = split_sequence_overlap(features_test, targets_test, parameters.test_seq_dim, parameters.test_overlap_size)
    else:
      features_test, targets_test = split_sequence_nooverlap(features_test, targets_test, parameters.test_seq_dim, parameters.test_overlap_size)

    print(f"Number of training examples: {len(targets_train)}")
    print(f"Number of validation examples: {len(targets_val)}")
    print(f"Number of test examples: {len(targets_test)}")

    # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
    featuresTrain = torch.from_numpy(features_train)
    if regression:
      targetsTrain = torch.from_numpy(targets_train).type(torch.FloatTensor)  
    else:
      targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)  # data type is long

    featuresVal = torch.from_numpy(features_val)
    if regression:
      targetsVal = torch.from_numpy(targets_val).type(torch.FloatTensor)
    else:
      targetsVal = torch.from_numpy(targets_val).type(torch.LongTensor)  # data type is long

    # Pytorch train and validation sets
    train = TensorDataset(featuresTrain, targetsTrain)
    val = TensorDataset(featuresVal, targetsVal)
    
    if balance_classes:
      train_sampler = ImbalancedDatasetSampler(train)
      valid_sampler = ImbalancedDatasetSampler(val)
      # data loader
      train_loader = DataLoader(train, batch_size=parameters.batch_size, sampler=train_sampler)
      val_loader = DataLoader(val, batch_size=parameters.batch_size, sampler=valid_sampler)
    else:
      # data loader
      train_loader = DataLoader(train, batch_size=parameters.batch_size, shuffle=True)
      val_loader = DataLoader(val, batch_size=parameters.batch_size, shuffle=False)


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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    cur_learning_rate = learning_rate
    #scheduler = lrs.ExponentialLR(optimizer, gamma=0.95)



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


    loss_list = []
    val_loss_list = []
    epoch_list = []
    accuracy_list = []
    val_accuracy_list = []
    count = 0
    min_val_loss = 1e6
    if regression:
      max_val_accuracy = 1e8
    else:
      max_val_accuracy = 0
    epoch = 0
    all_predicted = []
    all_labels = []
    all_outputs = np.empty((0, nclasses), dtype='float')
    rest_patience = patience

    #for epoch in range(num_epochs):
    while rest_patience>0:
        all_train_predicted = []
        all_train_labels = []
        all_train_outputs = np.empty((0, nclasses), dtype='float')

        train_loss = 0
        for i, (features, labels) in enumerate(train_loader):

            model.train()

            train = Variable(features.view(-1, parameters.seq_dim, input_dim)).to(device)
            labels = Variable(labels).to(device)

            # Data augmentation
            if data_augmentation:
              noise = gaussian_noise_std * torch.randn(train.shape[0], parameters.seq_dim, floatcols.sum())
              train[:,:, floatcols] += noise.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(train)
            # Calculate softmax and cross entropy loss
            if regression and outputs.ndim>1:
              loss = error(torch.squeeze(outputs), labels)
            else:
              loss = error(outputs, labels)
            train_loss += labels.shape[0] * loss

            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            predicted = torch.max(outputs.data, 1)[1]
            predicted = predicted.to('cpu')
            all_train_predicted.extend(list(predicted.detach().numpy()))
            all_train_labels.extend(list(labels.cpu().detach().numpy()))
            all_train_outputs = np.concatenate((all_train_outputs, outputs.data.to('cpu').reshape(-1, nclasses)))

            count += 1
            # end of training epoch
        train_loss = train_loss.detach().cpu().numpy() / len(targets_train)



        # Calculate validation accuracy
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

            if regression and outputs.ndim>1:
              val_loss = error(torch.squeeze(outputs), labels)
            else:
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

        al_np = np.array(all_val_labels)   
        ao_np = np.array(all_val_outputs)  
        if regression:
          if ao_np.ndim>1:
            ao_np2 = np.squeeze(ao_np)
          else:
            ao_np2 = ao_np
          accuracy = mean_squared_error(al_np, ao_np2)
        else: # classification
          accuracy = correct / float(total)

        # store loss and iteration
        loss_list.append(loss.data)
        val_loss_list.append(val_loss.data)
        epoch_list.append(epoch)
        accuracy_list.append(accuracy)
        print('Subject: {}/{}  Epoch: {:>3}  Loss: {:.6}/{:.6}  Validation accuracy: {:.2f}'.format(test_subj, xv, epoch, train_loss, val_loss, accuracy))

        if epoch==0:
          init_val_loss = val_loss

        #if val_loss<min_val_loss:
        if accuracy>xv_max_val:
          #print("    Saving model to", model_filename)
          #torch.save(model.state_dict(), model_filename)
          xv_max_val = accuracy
        if regression:
          early_stoppying_condition = (accuracy<max_val_accuracy)
        else: # classification
          early_stoppying_condition = (accuracy>max_val_accuracy)
        if early_stoppying_condition:
          min_val_loss = val_loss
          max_val_accuracy = accuracy
          #print(f"    Max validation for this run. Current max: {xv_max_val}")
          print(f"    Max validation for this run. Current max: {max_val_accuracy}")
          print("    Saving model to", model_filename)
          torch.save(model.state_dict(), model_filename)
          rest_patience = patience
          all_maxval_predicted = all_val_predicted
          all_maxval_labels = all_val_labels
          all_maxval_outputs = all_val_outputs
          all_maxtrain_predicted = all_train_predicted
          all_maxtrain_labels = all_train_labels
          all_maxtrain_outputs = all_train_outputs
        torch.save(model.state_dict(), "tmp.ckpt")

        if rest_patience+epoch<min_num_epochs:
          rest_patience = min_num_epochs-epoch
        else:
          rest_patience -= 1
        if reinit==True and rest_patience==0:
          #if min_val_loss>0.90*init_val_loss:
          if epoch<min_epoch_of_max_val:
            print("Reinitialising model...")
            model = LSTMModel3(input_dim, parameters.hidden_dim, parameters.layer_dim, nclasses, device)
            cur_learning_rate *= 0.5
            optimizer = torch.optim.Adam(model.parameters(), lr=cur_learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
            rest_patience = patience
            min_val_loss = 1e5
            if regression:
              max_val_accuracy = 1e8
            else:
              max_val_accuracy = 0
            loss_list = []
            val_loss_list = []
            epoch_list = []
            accuracy_list = []
            val_accuracy_list = []
            epoch=0

        epoch += 1
    #scheduler.step()
    copyfile(model_filename, f"{modeldir}/checkpoint_{test_subj:02}_{xv:02}.ckpt")

    val_acc_val_loss_list.append((max_val_accuracy, min_val_loss))


    # Test
    model.load_state_dict(torch.load(model_filename))
    correct = 0
    total = 0
    prev_label = -1
    class_hist = np.zeros(nclasses, dtype='int')
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
        
      else:
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
    if regression:
      if ao_np.ndim>1:
        ao_np2 = np.squeeze(ao_np)
      else:
        ao_np2 = ao_np
      accuracy = mean_squared_error(al_np, ao_np2)
    else: # classification
      accuracy = correct / float(total)

    print(f"Test accuracy for run {xv}: {accuracy}")

    avg_test_acc += accuracy
    test_acc_list.append(accuracy)

  # remove run with lowest validation accuracy
  if remove_lowest_run:
    min_vacc = 1.0
    max_vloss = 0
    i = 0
    eps = 1e-5
    for vacc,vloss in val_acc_val_loss_list:
      if vacc<min_vacc or (abs(vacc-min_vacc)<eps and vloss>max_vloss):
        del_idx = i
        min_vacc = vacc
        max_vloss = vloss
      i += 1
    val_acc_val_loss_list.pop(del_idx)
    test_acc_list.pop(del_idx)
  avg_test_acc = np.mean(test_acc_list)

  # avg_test_acc /= parameters.cross_validation_passes
  test_accuracies.append(avg_test_acc)
  print("Test accuracies:")
  print(test_accuracies)
  print(f"Mean accuracy: {np.mean(test_accuracies)}")

  # calibrate class probabilities
  '''
  sm_train_outputs = scipy.special.softmax(all_maxtrain_outputs, axis=1) # outputs of train set
  sm_val_outputs = scipy.special.softmax(all_maxval_outputs, axis=1) # outputs of val set
  sm_outputs = scipy.special.softmax(all_outputs, axis=1) # outputs of test set
  atl_np = np.array(all_maxtrain_labels)  # train
  avl_np = np.array(all_maxval_labels) # val
  al_np = np.array(all_labels)   # test
  # calibrate validation set on train outputs
  c0 = (atl_np==0).astype(int)
  c1 = (atl_np==1).astype(int)
  c2 = (atl_np==2).astype(int)
  p0, _ = VennABERS.ScoresToMultiProbs(list(zip(sm_train_outputs[:,0], c0)), sm_val_outputs[:,0])
  p1, _ = VennABERS.ScoresToMultiProbs(list(zip(sm_train_outputs[:,1], c1)), sm_val_outputs[:,1])
  p2, _ = VennABERS.ScoresToMultiProbs(list(zip(sm_train_outputs[:,2], c2)), sm_val_outputs[:,2])
  p012 = np.vstack((p0, p1, p2)) 
  sm_p012 = scipy.special.softmax(p012.T, axis=1)
  calibrated_pred = np.argmax(sm_p012, axis=1)
  calibrated_accuracy = (calibrated_pred==avl_np).sum()/len(avl_np)
  print(f"val acc: {max_val_accuracy}  cal: {calibrated_accuracy}")
  
  if calibrated_accuracy>max_val_accuracy:
    print("Using calibrated outputs.")
    #c0 = (atl_np==0).astype(int)
    #c1 = (atl_np==1).astype(int)
    #c2 = (atl_np==2).astype(int)
    p0, _ = VennABERS.ScoresToMultiProbs(list(zip(sm_train_outputs[:,0], c0)), sm_outputs[:,0])
    p1, _ = VennABERS.ScoresToMultiProbs(list(zip(sm_train_outputs[:,1], c1)), sm_outputs[:,1])
    p2, _ = VennABERS.ScoresToMultiProbs(list(zip(sm_train_outputs[:,2], c2)), sm_outputs[:,2])
    p012 = np.vstack((p0, p1, p2)) 
    sm_p012 = scipy.special.softmax(p012.T, axis=1)
    calibrated_pred = np.argmax(sm_p012, axis=1)
    calibrated_accuracy = (calibrated_pred==atl_np).sum()/float(total)
    calibrated_test_accuracies.append(calibrated_accuracy)
  else:
    calibrated_test_accuracies.append(accuracy)
    '''
  '''
  lr = LR()
  atl_np = np.array(all_maxtrain_labels)  # train
  avl_np = np.array(all_maxval_labels) # val
  tvl = np.concatenate((atl_np, avl_np)) 
  tvo = np.vstack((all_train_outputs, all_maxval_outputs)) 
  lr.fit(all_maxval_outputs, avl_np)  # fit on validation set
  #lr.fit(tvo, tvl)  # fit on train+val set
  pred = lr.predict_proba(all_outputs)
  mpred = np.argmax(pred, axis=1) 
  calibrated_accuracy = (all_labels==mpred).sum()/len(all_labels)
  print(f"val acc: {max_val_accuracy}  cal: {calibrated_accuracy}")
  
  if calibrated_accuracy>max_val_accuracy:
    print("Using calibrated outputs.")
    lr.fit(tvo, tvl)  # recompute fit on train+val set
    pred = lr.predict_proba(all_outputs)
    mpred = np.argmax(pred, axis=1) 
    calibrated_accuracy = (all_labels==mpred).sum()/len(all_labels)
    calibrated_test_accuracies.append(calibrated_accuracy)
  else:
    calibrated_test_accuracies.append(accuracy)
  
  print("Calibrated test accuracies:")
  print(calibrated_test_accuracies)
  print(f"Mean calibrated accuracy: {np.mean(calibrated_test_accuracies)}")
  '''

  if do_plots:
    # visualization loss
    plt.subplot(4,4, test_subj+1)
    plt.plot(epoch_list, loss_list, label='train')
    plt.plot(epoch_list, val_loss_list, label='val')
    #plt.xlabel("Number of iteration")
    #plt.ylabel("Loss")
    #plt.title(f"Loss vs Number of iteration: subject {test_subj}")
    #plt.legend()
    #plt.ion()
    #plt.show()
    #plt.pause(1)
    #plt.ioff()
    #FigureCanvasBase.flush_events()

    # visualization accuracy
    #plt.figure()
    #plt.plot(epoch_list, accuracy_list, color="red")
    #plt.xlabel("Number of iteration")
    #plt.ylabel("Accuracy")
    #plt.title("RNN: Accuracy vs Number of iteration")
    #plt.show()

if do_plots:
  plt.ion()
  plt.show()

