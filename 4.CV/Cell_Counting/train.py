import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data import FluoData
from torch.utils.data import DataLoader
from model import UNet, FCRN_A
from config import opt
from looper_d import Looper

def calc_errors(true_values, predicted_values, size):
    """
    Calculate errors and standard deviation based on current
    true and predicted values.
    """
    stats = {}
    err = np.array(true_values) - np.array(predicted_values)
    abs_err = np.abs(err)
    stats['mean_err'] = np.sum(err) / size
    stats['mean_abs_err'] = np.sum(abs_err) / size
    stats['std'] = err.std()
    return stats

def plot(plots, true_values, predicted_values, loss, valid):
    """Plot true vs predicted counts and loss."""
    # true vs predicted counts
    true_line = [[0, max(true_values)]] * 2  # y = x
    plots[0].cla()
    plots[0].set_title('Train' if not valid else 'Valid')
    plots[0].set_xlabel('True value')
    plots[0].set_ylabel('Predicted value')
    plots[0].plot(*true_line, 'r-')
    plots[0].scatter(true_values, predicted_values)

    # loss
    epochs = np.arange(1, len(loss) + 1)
    plots[1].cla()
    plots[1].set_title('Train' if not valid else 'Valid')
    plots[1].set_xlabel('Epoch')
    plots[1].set_ylabel('Loss')
    plots[1].plot(epochs, loss)

    plt.pause(0.01)
    plt.tight_layout()

def train():
    """Main training process."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if opt.data_type == 'bacteria':
        train_h5pth = opt.h5_path+'train.h5'
        val_h5pth = opt.h5_path+'valid.h5'
    elif opt.data_type == 'cell':
        train_h5pth = opt.cell_h5_path+'train.h5'
        val_h5pth = opt.cell_h5_path+'valid.h5'
    
    train_data = FluoData(train_h5pth, opt.data_type, color=opt.color, horizontal_flip=1.0 * opt.h_flip, vertical_flip=1.0 * opt.v_flip)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size)
    val_data = FluoData(val_h5pth, opt.data_type, color=opt.color, horizontal_flip=0, vertical_flip=0)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size)
    
    if opt.model.find("UNet") != -1:
        model = UNet(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    elif opt.model == "FCRN_A":
        model = FCRN_A(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    model = torch.nn.DataParallel(model)
    
    # if os.path.exists('{}.pth'.format(opt.model)):
    #     model.load_state_dict(torch.load('{}.pth'.format(opt.model)))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=0.9,
                                weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)
    
    # if plot flag is on, create a live plot (to be updated by Looper)
    if opt.plot:
        plt.ion()
        fig, plots = plt.subplots(nrows=2, ncols=2)
    else:
        plots = [None] * 2
    
    best_result = float('inf')
    best_epoch = 0

    train_looper = Looper(model, device, criterion, optimizer,
                          train_dataloader, len(train_data), plots[0])
    valid_looper = Looper(model, device, criterion, optimizer,
                          val_dataloader, len(val_data), plots[1],
                          validation=True)


    for epoch in range(opt.epochs):

        print("======= epoch {} =======".format(epoch))

        ###############################################
        ########         Training Phase        ########
        ###############################################
        train_looper.run()
        lr_scheduler.step()
        
        ###############################################
        ########       Validation Phase        ########
        ###############################################
        with torch.no_grad():
            result = valid_looper.run()

        if result < best_result:
            best_result = result
            best_epoch = epoch
            torch.save(model.state_dict(), '{}.pth'.format(opt.model))

            print(f"\nNew best result: {best_result}")

    print("[Training done] Best epoch: {}".format(best_epoch))
    print("[Training done] Best result: {}".format(best_result))

if __name__ == '__main__':

    train()