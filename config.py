import os




#Path to Data :
if os.uname().nodename == 'lx37-Kubiak': #'tkz-XPS'
    data_path = "/home/lx37/Projets/FPerrin2022_VideoComaEEG_Bruno/Comportement/Data_Video"
    data_raw_path = data_path + "/raw/"
    data_processed_path = data_path + "/OP_OF_processed/"
    output_path = data_path + "/after_loading/"
else:
    data_raw_path = "D:/Bruno/Raw/" 
    data_processed_path = "D:/Bruno/processed/"
    output_path =  "C:/Users/Bruno/Documents/MetaDossier/"
