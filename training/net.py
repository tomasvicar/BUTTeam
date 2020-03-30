import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init


class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=1,do_batch=1,dov=0):
        super().__init__()
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



class Net(nn.Module):
    def set_t(self,t):
        self.t=t
        
    def get_t(self):
        return self.t
    
    
    def __init__(self, levels=7,lvl1_size=2,input_size=12,output_size=9,convs_in_layer=3):
        super().__init__()
        self.levels=levels
        self.lvl1_size=lvl1_size
        self.input_size=input_size
        self.output_size=output_size
        self.convs_in_layer=convs_in_layer
        
        self.t=0.5*np.ones(output_size)
        
        
        
        
        self.layers=nn.ModuleList()
        for lvl_num in range(self.levels):
            
            if lvl_num==0:
                self.layers.append(myConv(input_size, int(lvl1_size*2**lvl_num)))
            else:
                self.layers.append(myConv(int(lvl1_size*2**(lvl_num-1)), int(lvl1_size*2**lvl_num)))
            
            for conv_num_in_lvl in range(self.convs_in_layer-1):
                self.layers.append(myConv(int(lvl1_size*2**lvl_num), int(lvl1_size*2**lvl_num)))

            self.fc=nn.Linear(int(lvl1_size*2**(self.levels-1)), self.output_size)
        
        
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
        
        
        
    def forward(self, x,lens):
        
        #tady dodělat vymazání na násobek bloku
        
        
        layer_num=-1
        for lvl_num in range(self.levels):
            
            
            for conv_num_in_lvl in range(self.convs_in_layer):
                layer_num+=1
                if conv_num_in_lvl==1:
                    y=x
                
                x=self.layers[layer_num](x)
                
            x=x+y
            x=F.max_pool1d(x, 2, 2)
            
            
            
        
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
        
        
        
        

                
        

