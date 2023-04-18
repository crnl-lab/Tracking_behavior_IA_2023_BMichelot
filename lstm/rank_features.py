from corankco.dataset import Dataset
from corankco.scoringscheme import ScoringScheme
from corankco.algorithms.algorithmChoice import get_algorithm
from corankco.algorithms.algorithmChoice import Algorithm
from corankco.kemeny_computation import KemenyComputingFactory
import scipy.stats
import csv
import numpy as np

data_filename = "/home/stefan/data/private/eveilcoma/temoins2022/Tables_DL_nofilter/All_Subs_Diff_Modules_nofilter_withoutAUc.csv"
with open(data_filename, newline='') as f:
  reader = csv.reader(f)
  labels = next(reader) 

labels.remove('Subject')
labels.remove('Condition')
labels.remove('Emotion')
labels.remove('Presence')

# groupe 'soc'
fa = np.loadtxt('feature_attributions/feature_attributions_1-5.csv', delimiter=',')
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_1-6.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_2-5.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_2-6.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_3-5.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_3-6.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_4-5.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_4-6.csv', delimiter=',')))

'''
# groupe 'emo'
fa = np.loadtxt('feature_attributions/feature_attributions_5-6.csv', delimiter=',')
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_1-2.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_3-4.csv', delimiter=',')))
'''

'''
# groupe 'stim'
fa = np.loadtxt('feature_attributions/feature_attributions_1-3.csv', delimiter=',')
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_1-4.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_2-3.csv', delimiter=',')))
fa = np.concatenate((fa, np.loadtxt('feature_attributions/feature_attributions_2-4.csv', delimiter=',')))
'''


ranks = scipy.stats.rankdata(-fa, method='average', axis=1)
ind = np.argsort(ranks, axis=1)
ind2 = np.expand_dims(ind, axis=2)

# Method 1: Rank aggregation by MedRank
dataset = Dataset(ind2)
sc = ScoringScheme()
# available aggregation algorithms: BioConsert, ParCons, ExactAlgorithm, KwikSortRandom,
#   RepeatChoice, PickAPerm, MedRank, BordaCount, BioCo, CopelandMethod
alg = get_algorithm(alg=Algorithm.MedRank)
consensus = alg.compute_consensus_rankings(dataset=dataset, scoring_scheme=sc, return_at_most_one_ranking=False)
medrank_feature_ranking = consensus.consensus_rankings  # final ranking

# Method 2: Rank aggregation by average 
avg_ranks = np.mean(ranks, axis=0)
avg_ind = np.argsort(avg_ranks) # final ranking
feature_ranking = [labels[i] for i in avg_ind] # convert the indices into feature names

