class Config:

    img_path = 'images/image3.jpg'

    # superpixel
    compactness = 100
    num_superpixels = 10000
    min_labels = 3

    # model
    nChannel = 200
    nConv = 2
    nClass = 100
    
    # training
    model_path = 'model.pth'
    num_epoch = 5
    lr = 0.09 # 0.1
    momentum = 0.9
    maxIter = 200 # 100
    lamda = 0.005
    

opt = Config()