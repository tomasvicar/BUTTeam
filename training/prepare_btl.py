  
import os
import json
import scipy.io as io
import numpy as np


path = "../../cinc2020_b"
label_file_name = "labels.json"

# data='filt'
for data in ['raw','filt']:

    path_save="../../cinc2020_b_cincformat_" + data
    
    try:
        os.mkdir(path_save)
    except:
        pass
    
    
    
    with open(os.path.join(path, label_file_name), "r") as file:
            labels = json.load(file)
    
    
    
    
    for key, value in labels.items():
        
        
        value_new=[]
        for pato in value:
            if pato=='ST':
                pato='STD'
            value_new.append(pato)
           
            
        value_new=np.unique(value_new)##někde tam bylo STE dvakrat
        
        
        name=os.path.join(path_save, key)+".hea"
        
        
        with open(name, "w") as file:
            for line_idx in range(16):
                
                if line_idx == 15:
                    file.write('#Dx: ' +','.join(value_new) )
                else:
                    file.write(key + ' ' + str(line_idx) +  " \n")  
        file.close()
            
        
        name_mat=os.path.join(path, key)+".mat"
        
       
        
        
        name_mat_save=os.path.join(path_save, key)+".mat"
        
        sample = io.loadmat(name_mat)
        sample=sample[data]
        sample=sample.astype(np.float64)
        
        io.savemat(name_mat_save,{'val':sample})
    
    
    
    