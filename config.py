class Config:
    models = {
        'mbert': 'bert-base-multilingual-cased',
        'xlmr': 'xlm-roberta-base',
        'phobert': 'vinai/phobert-base'
    }
    
    models_to_train = ['mbert', 'xlmr', 'phobert']
    
    data_path = '/kaggle/working/ViQuAD2.0'
    train_dir = 'train'
    dev_dir = 'dev'
    test_dir = 'test'
    
    output_dir = '/kaggle/working/output'
    
    max_length = 384
    doc_stride = 128
    max_answer_length = 30
    
    batch_size = 16
    learning_rate = 3e-5
    num_epochs = 3
    warmup_steps = 500
    weight_decay = 0.01
    
    early_stopping_patience = 3
    early_stopping_threshold = 0.001
    
    seed = 42
    fp16 = True

