class Config:
    
    
    best_models_dir='best_models'
    
    pato_names=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
    
    DATA_PATH = "../../Training_WFDB/"
    
    train_batch_size = 32
    train_num_workers=4
    valid_batch_size = 32
    valid_num_workers=4

    max_epochs = 107
    step_size=35
    gamma=0.1
    init_lr=0.01
    
    
    info_save_dir='data_split'
    split_ratio=[8,2]
    num_of_splits=10
    
    model_save_dir='../../tmp'
    
    model_note='no_pretrain'
    
    best_t=True
    
    loss_fcn='wce'
    
    
    pretrained=None
    # pretrained='model_name'
    
    levels=9
    lvl1_size=8
    input_size=12
    output_size=9
    convs_in_layer=6
    init_conv=8
    filter_size=13
    
    
    ploting=False
    
    
