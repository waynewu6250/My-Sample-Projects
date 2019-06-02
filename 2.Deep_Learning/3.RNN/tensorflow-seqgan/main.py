import torch as t
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import tqdm
import pickle
import re

from eval_model.data import get_dataloader
from eval_model.class_model import ClrModel
from eval_model.config import opt

def train(**kwargs):

    #Parameters
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    for k, v in kwargs.items():
        setattr(opt, k, v)

    #Data
    dataloader = get_dataloader(opt)

    #Model
    model = ClrModel(dataloader.dataset.word2id, dataloader.dataset.num_classes, opt)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model = model.to(device)

    optimizer = Adam(model.parameters(), opt.lr)
    criterion = t.nn.BCELoss()

    for epoch in range(opt.max_epoch):
        for (captions_t, captions_p, lengths_t, lengths_p) in dataloader:
            
            batch_size = captions_t.size(1)
            true_labels = t.ones(batch_size).unsqueeze(1).to(device)
            fake_labels = t.zeros(batch_size).unsqueeze(1).to(device)

            captions_t = Variable(captions_t).to(device)
            captions_p = Variable(captions_p).to(device)
            
            # Fake data
            captions_t_fake = captions_t.clone()
            np.random.shuffle(captions_t_fake)
            
            real = model(captions_t, lengths_t)
            fake = model(captions_t_fake, lengths_t)
            
            optimizer.zero_grad()

            loss_real = criterion(real, true_labels)
            loss_fake = criterion(fake, fake_labels)
            
            total_loss = loss_real+loss_fake
            
            total_loss.backward()
            optimizer.step()


        print("======Current Epoch: {}======".format(epoch))
        print("Current real Loss: ", total_loss.item())
        #print("Current fake Loss: ", loss_fake.item())

        if (epoch+1) % opt.save_model == 0:
            t.save(model.state_dict(), "checkpoints/{}.pth".format(epoch))
    

def evaluate(test_str = "60 tear old male, giant confluent drusen.",
             truth_str = "20 tear old male, srnv-md giant confluent drusen.",
             label = "srnv-md"):

    #Parameters
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    
    #Load Data
    with open(opt.data_path,'rb') as f:
        results = pickle.load(f)
    test_toks = [results['word2id'][tok] for tok in text_prepare(test_str)]
    test_toks = t.LongTensor(test_toks).unsqueeze(1)
    truth_toks = [results['word2id'][tok] for tok in text_prepare(truth_str)]
    truth_toks = t.LongTensor(truth_toks).unsqueeze(1)

    vocab_list = t.load(opt.vocab_path)
    reverse_vocab = {j:i for i,j in vocab_list.items()}

    #Load Model
    model = ClrModel(results['word2id'], opt.num_classes, opt)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model = model.to(device)
    
    # Predict
    pred1 = model.predict(test_toks)
    prediction1 = t.Tensor(pred1[1][0])

    pred2 = model.predict(truth_toks)
    prediction2 = t.Tensor(pred2[1][0])

    #ground_truth = t.Tensor(np.eye(opt.num_classes)[vocab_list[label]])
    cosine_similarity = t.dot(prediction1, prediction2)/(t.norm(prediction1)*t.norm(prediction2))
    
    score = cosine_similarity.item()
    pred_label = reverse_vocab[t.argmax(pred1[1]).item()]
    # print("Score: ",cosine_similarity.item())
    # print("Real Label: ",label)
    # print("Predicted Label: ",reverse_vocab[t.argmax(pred1[1]).item()])
    return score, pred_label, label
    



if __name__ == "__main__":
    #import fire
    #fire.Fire()
    train()
        
    




    