
import os
import glob
import numpy as np
import glob
import numpy as np
from torch import optim
from torch.utils import data as dataa
import torch
from shutil import copyfile,rmtree
from datetime import datetime
import logging
import sys


from utils.utils import get_lr
from utils.collate import PaddedCollate
from config import Config
from utils.log import Log
from utils.compute_challenge_metric_custom import compute_challenge_metric_custom
from utils.optimize_ts import optimize_ts,aply_ts
from dataset import Dataset
import net
from utils.utils import AdjustLearningRateAndLoss
from utils.get_stats import get_stats

from run_12ECG_classifier import run_12ECG_classifier,load_12ECG_model
from driver import load_challenge_data,save_challenge_predictions


# from evaluate_12ECG_score import evaluate_12ECG_score
# from evaluate_12ECG_score_fixed import evaluate_12ECG_score
from evaluate_12ECG_score_fixed_nan import evaluate_12ECG_score



def train_12ECG_classifier(input_directory, output_directory):
    for model_num,model_seed in enumerate(Config.MODELS_SEEDS):
        train_one_model(input_directory, output_directory,model_num,model_seed)
    
    
def train_one_model(input_directory, output_directory,model_num,model_seed):   
    device = Config.DEVICE
    
    file_list = glob.glob(input_directory + "/**/*.mat", recursive=True)

    num_files = len(file_list)
    print(num_files)

    lbl_counts,lens=get_stats(file_list)
      
    file_list2=[]
    for ind in range(len(file_list)):
        if lens[ind]<(Config.output_sampling * 125):
            file_list2.append(file_list[ind])
    
    
    file_list=file_list2

    num_of_sigs=num_files

    w_positive=num_of_sigs/lbl_counts
    w_negative=num_of_sigs/(num_of_sigs-lbl_counts)
    
    w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
    w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)
    
    num_files = len(file_list)

    # Train-Test split
    state=np.random.get_state()
    np.random.seed(model_seed)
    split_ratio_ind = int(np.floor(Config.SPLIT_RATIO[0] / (Config.SPLIT_RATIO[0] + Config.SPLIT_RATIO[1]) * num_files))
    permuted_idx = np.random.permutation(num_files)
    train_ind = permuted_idx[:split_ratio_ind]
    valid_ind = permuted_idx[split_ratio_ind:]
    partition = {"train": [file_list[file_idx] for file_idx in train_ind],
        "valid": [file_list[file_idx] for file_idx in valid_ind]}
    np.random.set_state(state)


    
    # Train dataset generator
    training_set = Dataset( partition["train"],transform=Config.TRANSFORM_DATA_TRAIN,encode=Config.TRANSFORM_LBL)
    training_generator = dataa.DataLoader(training_set,batch_size=Config.BATCH_TRAIN,num_workers=Config.TRAIN_NUM_WORKERS,
                                         shuffle=True,drop_last=True,collate_fn=PaddedCollate() )
    
    
    validation_set = Dataset(partition["valid"],transform=Config.TRANSFORM_DATA_VALID,encode=Config.TRANSFORM_LBL)
    validation_generator = dataa.DataLoader(validation_set,batch_size=Config.BATCH_VALID,num_workers=Config.VALID_NUM_WORKERS,
                                           shuffle=False,drop_last=False,collate_fn=PaddedCollate() )
    
    
    model = net.Net_addition_grow(levels=Config.LEVELS,
                                  lvl1_size=Config.LVL1_SIZE,
                                  input_size=Config.INPUT_SIZE,
                                  output_size=Config.OUTPUT_SIZE,
                                  convs_in_layer=Config.CONVS_IN_LAYERS,
                                  init_conv=Config.INIT_CONV,
                                  filter_size=Config.FILTER_SIZE)
    
    model.save_train_names(partition)
    model=model.to(device)

    ## create optimizer and learning rate scheduler to change learnng rate after 
    optimizer = optim.Adam(model.parameters(),lr =Config.LR_LIST[0] ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=AdjustLearningRateAndLoss(optimizer,Config.LR_LIST,Config.LR_CHANGES_LIST,Config.LOSS_FUNTIONS)
    
    log=Log(['loss','challange_metric'])
    
    for epoch in range(Config.MAX_EPOCH):
        
        print(get_lr(optimizer))
        print(scheduler.actual_loss)
        
        
        #change model to training mode
        model.train()
        N=len(training_generator)
        lens_all=[]
        for it,(pad_seqs,lbls,lens) in enumerate(training_generator):
            if it%10==0:
                print(str(it) + '/' + str(N))
            
            ## send data to graphic card
            pad_seqs,lens,lbls = pad_seqs.to(device),lens.to(device),lbls.to(device)

            ## aply model
            res=model(pad_seqs,lens)
            
            ## calculate loss
            loss=scheduler.actual_loss(res,lbls,w_positive_tensor,w_negative_tensor)

            ## update model 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()
            lens=lens.detach().cpu().numpy()

            lens_all.append(lens)
            
            challange_metric=compute_challenge_metric_custom(res>0.5,lbls)

            ## save results
            log.append_train([loss,challange_metric])
                       

        model.save_lens(np.concatenate(lens_all,axis=0))
        ## validation mode - "disable" batch norm 
        res_all=[]
        lbls_all=[]
        model.eval() 
        N=len(validation_generator)
        for it,(pad_seqs,lbls,lens) in enumerate(validation_generator):
            if it%10==0:
                print(str(it) + '/' + str(N))

            pad_seqs,lens,lbls = pad_seqs.to(device),lens.to(device),lbls.to(device)

            res=model(pad_seqs,lens)
            
            loss=scheduler.actual_loss(res,lbls,w_positive_tensor,w_negative_tensor)

            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            
            challange_metric=compute_challenge_metric_custom(res>0.5,lbls)

            ## save results
            log.append_test([loss,challange_metric])
            
            
            lbls_all.append(lbls)
            res_all.append(res)
                
        ts,opt_challenge_metric=optimize_ts(np.concatenate(res_all,axis=0),np.concatenate(lbls_all,axis=0)) 
        model.set_ts(ts)
        log.save_opt_challange_metric_test(opt_challenge_metric)
        
        log.save_and_reset()
        
        lr=get_lr(optimizer)
        
        info='model' + str(model_num) + '_'  + str(epoch) + '_' + str(lr) + '_train_'  + str(log.train_log['challange_metric'][-1]) + '_valid_' + str(log.test_log['challange_metric'][-1]) + '_validopt_' + str(log.opt_challange_metric_test[-1])
        print(info)
        
        model_name=output_directory+ os.sep + Config.MODEL_NOTE + info  
        log.save_log_model_name(model_name + '.pt')
        model.save_log(log)
        model.save_config(Config)
        torch.save(model,model_name + '.pt')
            
        log.plot(model_name)
        
        scheduler.step()
        
        
    best_model_name=log.model_names[np.argmax(log.opt_challange_metric_test)]
    copyfile(best_model_name,output_directory +'/model' + str(model_num)  + '.pt')
    
    
    
    
    

if __name__ == '__main__':
    logging.basicConfig(filename='debug.log',level=logging.INFO)
    try:
        # Parse arguments.
        input_directory = '../data_nofold'
        output_directory = 'model'
    
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
            
    
        print('Running training code...')
    
        train_12ECG_classifier(input_directory, output_directory)
    
        print('Done.')
        
    
    
    
    
    
        model_input = 'model'
        input_directory = '../data_nofold'
        output_directory = '../results'
        
        
    
        # Find files.
        input_files = []
        for f in os.listdir(input_directory):
            if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
                input_files.append(f)
    
    
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
    
        # Load model.
        print('Loading 12ECG model...')
        model = load_12ECG_model(model_input)
    
        # Iterate over files.
        print('Extracting 12ECG features...')
        num_files = len(input_files)
    
        for i, f in enumerate(input_files):
            print('    {}/{}...'.format(i+1, num_files))
            tmp_input_file = os.path.join(input_directory,f)
            data,header_data = load_challenge_data(tmp_input_file)
            current_label, current_score,classes = run_12ECG_classifier(data,header_data, model,traning_to_nan=True,file_name=f)
            # Save results.
            save_challenge_predictions(output_directory,f,current_score,current_label,classes)
    
    
        print('Done.')
    
    
    
    
        print('evaluating')
        auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric=evaluate_12ECG_score(input_directory, output_directory)
        
        print(challenge_metric)
        
        output_file='notes/result' + datetime.now().strftime("%H_%M_%d_%m_%Y") + '.txt'
        with open(output_file, 'w') as f:
            f.write(str(challenge_metric))
    except Exception as e:
        logging.critical(e, exc_info=True)