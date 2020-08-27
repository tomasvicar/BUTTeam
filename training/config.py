class Config:
    
    
    best_models_dir='best_models'
    
    pato_names=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
    
    DATA_PATH = "../../Training_WFDB/"
    # DATA_PATH ="../../train_split_tmp"
    
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
    num_of_splits=5
    
    model_save_dir='../../tmp'
    
    model_note='train_test_split'
    
    best_t=True
    
    loss_fcn='wce'
    
    
    # pretrained='best_models/pretraining0___277_0.001_train_0.7466433_valid_0.5399519.pkl'
    pretrained=None
    
    ## network setting
    levels=8
    lvl1_size=4
    input_size=12
    output_size=9
    convs_in_layer=2
    init_conv=4
    filter_size=13
    
    
    ploting=False
    
    
    
    
    
    
    
    #####pretrain
    
    
    # best_models_dir='best_models'
    
    # pato_names=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
    
    # DATA_PATH = "../../cinc2020_b_cincformat_filt/"
    
    # train_batch_size = 32
    # train_num_workers=4
    # valid_batch_size = 32
    # valid_num_workers=4

    # max_epochs = 107*4
    # step_size=35*4
    # gamma=0.1
    # init_lr=0.01
    
    
    # info_save_dir='data_split'
    # split_ratio=[8,2]
    # num_of_splits=1
    
    # model_save_dir='../../tmp'
    
    # model_note='pretraining'
    
    # best_t=True
    
    # loss_fcn='wce'
    
    
    # pretrained=None
    # # pretrained='model_name'
    
    # levels=9
    # lvl1_size=8
    # input_size=12
    # output_size=9
    # convs_in_layer=6
    # init_conv=8
    # filter_size=13
    
    
    # ploting=False
