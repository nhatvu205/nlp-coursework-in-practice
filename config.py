class Config:
    models = {
        'mbert': 'bert-base-multilingual-cased',
        'xlmr': 'xlm-roberta-base',
        'roberta': 'roberta-base'
    }
    
    models_to_train = ['mbert', 'xlmr', 'roberta']
    
    data_path = '/content/ViQuAD2.0'
    train_dir = 'train'
    dev_dir = 'dev'
    test_dir = 'test'
    
    output_dir = '/content/output'
    
    max_length = 384
    doc_stride = 128
    max_answer_length = 30
    
    batch_size = 16
    learning_rate = 3e-5
    num_epochs = 4
    warmup_steps = 500
    weight_decay = 0.01
    
    early_stopping_patience = 3
    early_stopping_threshold = 0.001
    
    seed = 42
    fp16 = True
