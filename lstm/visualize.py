import matplotlib.pyplot as plt
import numpy as np

def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):                            
  print(title)                                              
  for i in range(len(feature_names)):                        
    print(feature_names[i], ": ", '%.3f'%(importances[i]))                
  x_pos = (np.arange(len(feature_names)))
  if plot: 
    plt.figure(figsize=(12,6))
    plt.bar(x_pos, importances, align='center')                                
    plt.xticks(x_pos, feature_names, wrap=True)                         
    plt.xlabel(axis_title)
    plt.title(title)


def visualize_importances_max(feature_names, importances, firstn, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    ordered_list = []
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
        ordered_list.append((feature_names[i], importances[i]))
    ordered_list = sorted(ordered_list, key=lambda x: abs(x[1]), reverse=True)
    x_pos = np.arange(firstn)
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, [abs(y[1]) for y in ordered_list[0:firstn]], align='center')
        plt.xticks(x_pos, [y[0] for y in ordered_list[0:firstn]], wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)

#if do_plots:
  #plt.ioff()
  ##plt.figure()
  #plt.clf()
  #plt.subplot(441)
