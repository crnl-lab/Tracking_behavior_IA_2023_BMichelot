def initialize_parameters():
  global normalise_individual_subjects
  normalise_individual_subjects = False
  global seq_dim
  seq_dim= 1000       # length of sequences in frames
  global test_seq_dim
  test_seq_dim= 1100       # length of test sequences in frames
  global overlap_size
  overlap_size = 60  # number of frame of overlap in sequences
  global test_overlap_size
  test_overlap_size = 10
  global batch_size
  batch_size = 4# 16
  #global input_dim
  #input_dim = 101 #input dimension 
  global hidden_dim
  hidden_dim = 512  # hidden layer dimension
  global layer_dim
  layer_dim = 1  # number of hidden layers
  global test_with_subsequences
  test_with_subsequences = True
  global test_use_max
  test_use_max = False
  global cross_validation_passes
  cross_validation_passes=3
  
