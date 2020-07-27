import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt
import os





class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=0):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv1d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm1d(out_size,momentum=0.1)
        
        
        if self.dov>0:
            self.do=nn.Dropout(dov)
    
    def forward(self, inputs):
     
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        
        outputs=F.relu(outputs)
        
        if self.dov>0:
            outputs = self.do(outputs)
        
        return outputs


        
class Net_addition_grow(nn.Module):
    def set_ts(self,ts):
        self.ts=ts
        
    def get_ts(self):
        return self.ts
    
    
    def __init__(self, levels=7,lvl1_size=4,input_size=12,output_size=9,convs_in_layer=3,init_conv=4,filter_size=13,):
        super().__init__()
        self.levels=levels
        self.lvl1_size=lvl1_size
        self.input_size=input_size
        self.output_size=output_size
        self.convs_in_layer=convs_in_layer
        self.filter_size=filter_size
        
        
        
        self.init_conv=myConv(input_size,init_conv,filter_size=filter_size)
        
        
        self.layers=nn.ModuleList()
        for lvl_num in range(self.levels):
            
            
            if lvl_num==0:
                self.layers.append(myConv(init_conv, int(lvl1_size*(lvl_num+1)),filter_size=filter_size))
            else:
                self.layers.append(myConv(int(lvl1_size*(lvl_num))+int(lvl1_size*(lvl_num))+init_conv, int(lvl1_size*(lvl_num+1)),filter_size=filter_size))
            
            for conv_num_in_lvl in range(self.convs_in_layer-1):
                self.layers.append(myConv(int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1)),filter_size=filter_size))


        self.conv_final=myConv(int(lvl1_size*(self.levels))+int(lvl1_size*(self.levels))+init_conv, int(lvl1_size*self.levels),filter_size=filter_size)
        
        self.fc=nn.Linear(int(lvl1_size*self.levels), self.output_size)
        
        
        
        
        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
        
        
        
    def forward(self, x,lens):
        
        
        ## make signal len be divisible by 2**number of levels 
        ## replace rest by zeros
        for signal_num in range(list(x.size())[0]):
            
            k=int(np.floor(lens[signal_num].cpu().numpy()/(2**(self.levels-1)))*(2**(self.levels-1)))
            
            x[signal_num,:,k:]=0
        

        ## pad with more zeros  -  add as many zeros as convolution of all layers can proppagete numbers
        n=(self.filter_size-1)/2
        padded_length=n
        for p in range(self.levels):
            for c in range(self.convs_in_layer):
                padded_length=padded_length+2**p*n
        padded_length=padded_length+2**p*n+256 # 256 for sure

        
        shape=list(x.size())
        xx=torch.zeros((shape[0],shape[1],int(padded_length)),dtype=x.dtype)
        cuda_check = x.is_cuda
        if cuda_check:
            cuda_device = x.get_device()
            device = torch.device('cuda:' + str(cuda_device) )
            xx=xx.to(device)
        
        x=torch.cat((x,xx),2)### add zeros to signal
        
        x.requires_grad=True
        
        
        x=self.init_conv(x)
        
        x0=x
        
        ## aply all convolutions
        layer_num=-1
        for lvl_num in range(self.levels):
            
            
            for conv_num_in_lvl in range(self.convs_in_layer):
                layer_num+=1
                if conv_num_in_lvl==1:
                    y=x
                
                x=self.layers[layer_num](x)
                
            ## skip conection to previous layer and to the input
            x=torch.cat((F.avg_pool1d(x0,2**lvl_num,2**lvl_num),x,y),1)
            
            x=F.max_pool1d(x, 2, 2)
            
            
            
        x=self.conv_final(x)
        
        ### replace padded parts of signals by -inf => it will be not used in poolig
        for signal_num in range(list(x.size())[0]):
            
            k=int(np.floor(lens[signal_num].cpu().numpy()/(2**(self.levels-1))))
            
            x[signal_num,:,k:]=-np.Inf
            
        
        
        x=F.adaptive_max_pool1d(x,1)
        
        
        # N,C,1
        
        x=x.view(list(x.size())[:2])
        
        # N,C
        
        x=self.fc(x)
        
        x=torch.sigmoid(x)
        
        return x   
    
    def save_log(self,log):
        self.log=log
        
    def save_config(self,config):  
        self.config=config
        
    def save_lens(self,lens):
        self.lens=lens
        
    def save_train_names(self,train_names):
        
        tmp=[]
        for name in train_names['train']:
            path,filename = os.path.split(name)
            tmp.append(filename)
        train_names['train']=tmp
        
        tmp=[]
        for name in train_names['valid']:
            path,filename = os.path.split(name)
            tmp.append(filename)
        train_names['valid']=tmp
        
        self.train_names=train_names
        
        
        
        