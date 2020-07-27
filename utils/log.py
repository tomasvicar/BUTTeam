import numpy as np
import matplotlib.pyplot as plt


class Log():
    def __init__(self,names=['loss','acc']):
        
        self.names=names
        
        self.model_names=[]
            
        
        self.train_log=dict(zip(names, [[]]*len(names)))
        self.test_log=dict(zip(names, [[]]*len(names)))
        
        self.train_log_tmp=dict(zip(names, [[]]*len(names)))
        self.test_log_tmp=dict(zip(names, [[]]*len(names)))
        
        self.opt_challange_metric_test=[]
        
    def save_opt_challange_metric_test(self,opt_challange_metric):
        self.opt_challange_metric_test.append(opt_challange_metric)
        
        
    def append_train(self,list_to_save):
        for value,name in zip(list_to_save,self.names):
            self.train_log_tmp[name]=self.train_log_tmp[name] + [value]
        
        
    def append_test(self,list_to_save):
        for value,name in zip(list_to_save,self.names):
            self.test_log_tmp[name]=self.test_log_tmp[name] + [value]
        
        
    def save_and_reset(self):
        
        
        for name in self.names:
            self.train_log[name]= self.train_log[name] + [np.mean(self.train_log_tmp[name])]
            self.test_log[name]= self.test_log[name] + [np.mean(self.test_log_tmp[name])]
        
        
        
        self.train_log_tmp=dict(zip(self.names, [[]]*len(self.names)))
        self.test_log_tmp=dict(zip(self.names, [[]]*len(self.names)))
        
        
        
    def plot(self,save_name=None):
        if save_name is not None:
            save_names=[save_name,None]
        else:
            save_names=[None]
        
        for save_name in save_names:
            for name in self.names:
                plt.plot( self.train_log[name], label = 'train')
                plt.plot(self.test_log[name], label = 'test')
                plt.title(name)
                if save_name:
                    plt.savefig(save_name + name + '.png')
                plt.show()
                plt.close()
                
            name='opt_metric'
            plt.plot(self.opt_challange_metric_test, label = 'test')
            plt.title(name)
            if save_name:
                plt.savefig(save_name  + name + '.png' )
            plt.show()  
            plt.close()
            
            
            
    def save_log_model_name(self,model_name):
        ## store model names
        self.model_names.append(model_name)
        
        
        
    