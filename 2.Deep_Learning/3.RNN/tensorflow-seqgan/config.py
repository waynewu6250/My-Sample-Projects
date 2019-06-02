class Config:

    data_path = 'eval_model/results.pkl'
    pred_path = 'eval_model/results_vgg16.pkl'

    model_path = None #'eval_model/checkpoints/19.pth'
    vocab_path = 'eval_model/vocab_list.pth'
    batch_size = 32
    embedding_dim = 300
    hidden_dim = 256
    num_layers = 2
    max_epoch = 1000
    lr = 0.001
    num_classes = 47

    save_model = 5


    


opt = Config()