import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data import FluoData
from torch.utils.data import DataLoader
from model import UNet, FCRN_A
from config import opt

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

    train_data = FluoData(opt.h5_path+'train.h5', horizontal_flip=1.0 * opt.h_flip, vertical_flip=1.0 * opt.v_flip)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size)
    val_data = FluoData(opt.h5_path+'valid.h5', horizontal_flip=0, vertical_flip=0)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size)

    if opt.model == "UNet":
        model = UNet(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    else:
        model = FCRN_A(input_filters=3, filters=opt.unet_filters, N=opt.conv).to(device)
    model = torch.nn.DataParallel(model)

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

    for epoch in range(opt.epochs):

        print("======= epoch {} =======".format(epoch))

        ###############################################
        ########         Training Phase        ########
        ###############################################
        train_loss = []
        true_values = []
        predicted_values = []

        for img, label in train_dataloader:

            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            out = model(img)
            loss = criterion(out, label)
            train_loss.append(img.shape[0] * loss.item() / len(train_data))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            for true, predicted in zip(label, out):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 100 * no. of objects
                #       to make network learn better
                true_counts = torch.sum(true).item() / 100
                predicted_counts = torch.sum(predicted).item() / 100

                # update current epoch results
                true_values.append(true_counts)
                predicted_values.append(predicted_counts)

        stats = calc_errors(true_values, predicted_values, len(train_data))

        if opt.plot:
            plot(plots, true_values, predicted_values, train_loss, False)
        
        print("Training:\t Average loss: {:3.4f}\n \
               Mean error: {:3.3f}\n \
               Mean absolute error: {:3.3f}\n \
               Error deviation: {:3.3f}".format(train_loss[-1], stats['mean_abs_err'],  stats['std']))
        
        ###############################################
        ########       Validation Phase        ########
        ###############################################
        val_loss = []
        val_true_values = []
        val_predicted_values = []

        for img, label in val_dataloader:

            img = img.to(device)
            label = label.to(device)

            out = model(img)
            loss = criterion(out, label)
            val_loss.append(img.shape[0] * loss.item() / len(val_data))

            for true, predicted in zip(label, out):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 100 * no. of objects
                #       to make network learn better
                true_counts = torch.sum(true).item() / 100
                predicted_counts = torch.sum(predicted).item() / 100

                # update current epoch results
                val_true_values.append(true_counts)
                val_predicted_values.append(predicted_counts)

        val_stats = calc_errors(val_true_values, val_predicted_values, len(train_data))

        if opt.plot:
            plot(plots, val_true_values, val_predicted_values, val_loss, True)
        
        print("Validation:\t Average loss: {:3.4f}\n \
               Mean error: {:3.3f}\n \
               Mean absolute error: {:3.3f}\n \
               Error deviation: {:3.3f}".format(val_loss[-1], val_stats['mean_abs_err'],  val_stats['std']))

        if val_stats['mean_abs_err'] < best_result:
            best_result = val_stats['mean_abs_err']
            torch.save(model.state_dict(), '{}.pth'.format())

            print("\nNew best result: {}".format(val_stats['mean_abs_err']))

        print("\n", "-"*80, "\n", sep='')

    print("[Training done] Best result: {}".format(best_result))





