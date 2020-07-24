import numpy as np
import matplotlib.pyplot as plt


class Log():
    def __init__(self,names=['loss','acc']):
        
        self.names=names
        
        self.model_names=[]
            
        
        self.train_logs=dict(zip(names, [[]]*len(names)))
        self.test_logs=dict(zip(names, [[]]*len(names)))
        
        self.train_log_tmp=dict(zip(names, [[]]*len(names)))
        self.test_log_tmp=dict(zip(names, [[]]*len(names)))


        
    def append_train(self,list_to_save):
        
        for value,name in zip(list_to_save,self.names):
            self.train_log_tmp[name].append(value)
        
        
    def append_test(self,list_to_save):
        for value,name in zip(list_to_save,self.names):
            self.test_log_tmp[name].append(value)
        
        
    def save_and_reset(self):
        
        
        for name in self.names:
            self.train_log[name].append(np.mean(self.trainig_log_tmp[name]))
            self.test_log[name].append(np.mean(self.test_log_tmp[name]))
        
        
        self.train_log_tmp=dict(zip(self.names, [[]]*len(self.names)))
        self.test_log_tmp=dict(zip(self.names, [[]]*len(self.names)))
        
        
        
    def plot(self,save_name=None):
        
        for name in self.names:
            plt.plot( self.train_log, label = 'train')
            plt.plot(self.test_log, label = 'test')
            plt.title(name)
            if save_name:
                plt.savefig(save_name)
            plt.show()
            
            
    def save_log_model_name(self,model_name):
        ## store model names
        self.model_names.append(model_name)