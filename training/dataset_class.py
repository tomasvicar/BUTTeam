from torch.utils import data
from read_file import read_data
from read_file import read_lbl
import numpy as np
import torch 


class Dataset(data.Dataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, list_of_ids, data_path,split):
        """Initialization"""
        self.path = data_path
        self.list_of_ids = list_of_ids
        self.split=split
        
        self.MEANS=np.load('data_split/MEANS.npy')
    
        self.STDS=np.load('data_split/STDS.npy')
        
        self.pato_names=np.load('data_split/pato_names.npy')

    def __len__(self):
        """Return total number of data samples"""
        return len(self.list_of_ids)

    def __getitem__(self, idx):
        """Generate data sample"""
        # Select sample
        file_name = self.list_of_ids[idx]

        # Read data and get label
        X = read_data(self.path, file_name)
        
        
        
        sig_len=X.shape[1]
        signal_num=X.shape[0]
        
        if self.split=='train':
            if torch.rand(1).numpy()[0]>0.3:
                
                
                shift=torch.randint(sig_len,(1,1)).view(-1).numpy()
                
                X=np.roll(X, shift, axis=1)
                
                
            if torch.rand(1).numpy()[0]>0.3:
                
                max_resize_change=0.2
                relative_change=1+torch.rand(1).numpy()[0]*2*max_resize_change-max_resize_change
                new_len=int(relative_change*sig_len)
                
                Y=np.zeros((signal_num,new_len))
                for k in range(signal_num):

                    Y[k,:]=np.interp(np.linspace(0, sig_len-1, new_len),np.linspace(0, sig_len-1, sig_len),X[k,:])
                X=Y
                
            if torch.rand(1).numpy()[0]>0.3:
                
                max_mult_change=0.2
                
                for k in range(signal_num):
                    mult_change=1+torch.rand(1).numpy()[0]*2*max_mult_change-max_mult_change
                    X[k,:]=X[k,:]*mult_change
                    
                
                
                
                
        
        X=(X-self.MEANS.reshape(-1,1))/self.STDS.reshape(-1,1)
        
        
        lbl = read_lbl(self.path, file_name)
        
        y=np.zeros((len(self.pato_names),1)).astype(np.float32)
        
        lbl=lbl.split(',')
    
        for kk,p in enumerate(self.pato_names):
            for lbl_i in lbl:
                if lbl_i.find(p)>-1:
                    y[kk]=1
                    
                    
#        X=torch.from_numpy(X)
#        y=torch.from_numpy(y)

        return X,y
    
    def collate_fn(data):
        
        pad_val=0
        
        seqs, lbls = zip(*data)
        
        lens = [seq.shape[1] for seq in seqs]
        
        
        padded_seqs =pad_val*np.ones((len(seqs),seqs[0].shape[0], np.max(lens))).astype(np.float32)
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i,:, :end] = seq
        
        lbls=np.stack(lbls,axis=0)
        lbls=lbls.reshape(lbls.shape[0:2])
        
        lens = np.array(lens).astype(np.float32)
        
        padded_seqs=torch.from_numpy(padded_seqs)
        lbls=torch.from_numpy(lbls)
        lens=torch.from_numpy(lens)
        
        return padded_seqs,lens,lbls
        


def main():
    return Dataset


if __name__ == "__main__":
    main()
