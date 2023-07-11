from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from visualize import visualize_importances, visualize_importances_max


ap_np = np.array(all_predicted)
ig = IntegratedGradients(model)

features = featuresTest[targetsTest.detach().numpy()==ap_np]
targets = targetsTest[targetsTest.detach().numpy()==ap_np]
feature_dim = features.shape[2]

all_attr = np.empty((0, feature_dim), dtype='float')
for i in range(0, targets.shape[0], batch_size):
  end_index = min(i+batch_size, targets.shape[0])
  attr, delta = ig.attribute(features[i:end_index], target=targets[i:end_index], return_convergence_delta=True)
  attr_np = attr.detach().numpy()
  attr_np_mean = np.mean(attr_np, axis=1)  # mean over temporal window
  all_attr = np.vstack((all_attr, attr_np_mean))

#visualize_importances(list(train.columns), np.mean(all_attr, axis=0))
visualize_importances_max(list(train.columns), np.mean(all_attr, axis=0))

plt.ion()
plt.show()
