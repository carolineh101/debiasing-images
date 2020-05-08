import argparse
import numpy as np
import os
import pdb
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import *
from model import BaselineModel, OurModel
from utils import *

def main():
    pdb.set_trace()
    # Model Hyperparams
    hidden_size = opt.hidden_size
    learning_rate = opt.learning_rate
    save_after_x_epochs = 10

    # Determine device
    device = getDevice(opt.gpu_id)

    # Create data loaders
    data_loaders = load_celeba(splits=['train', 'valid'], batch_size=opt.batch_size, subset_size=31)
    train_data_loader = data_loaders['train']
    dev_data_loader = data_loaders['valid']

    # Create model
    model = BaselineModel(hidden_size)

    # Convert device
    model = model.to(device)

    # Create optimizer
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = learning_rate)

    start_epoch = 0
    best_bleu = 0.0
    train_batch_count = len(train_data_loader)

    #Load if resume
    checkpoint = None
    if opt.weights != '':
        checkpoint = torch.load(opt.weights, map_location=device)
        model.load_state_dict(checkpoint['model'])    
        if opt.resume:
            if checkpoint['epoch']:
                start_epoch = checkpoint['epoch'] + 1
            if checkpoint['best_bleu']:
                best_bleu = checkpoint['best_bleu']
            if checkpoint['optimizer']:
                optimizer.load_state_dict(checkpoint['optimizer']) 

    # Train loop
    pdb.set_trace()
    for epoch in range(start_epoch, opt.num_epochs):

        # Set model to train mode
        model.train()

        with tqdm(enumerate(train_data_loader), total=train_batch_count) as pbar: # progress bar
            for i, (images, targets) in pbar:

                # Shape: torch.Size([batch_size, 3, crop_size, crop_size])
                images = Variable(images.to(device))
                targets = Variable(targets.to(device))

                # Zero out buffers
                model.zero_grad()


                # Forward pass
                outputs = model(images)
                targets = targets.type_as(outputs)

                # CrossEntropyLoss is expecting:
                # Input:  (N, C) where C = number of classes
                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                s = ('%10s Loss: %.4f') % ('%g/%g' % (epoch, opt.num_epochs - 1), loss.item())
                pbar.set_description(s)

        # end batch ------------------------------------------------------------------------------------------------

        # Evaluate
        pdb.set_trace()
        model.eval()

        dev_batch_count = len(dev_data_loader.dataset)
        gt = Variable(torch.FloatTensor().to(device))
        pred = Variable(torch.FloatTensor().to(device))

        with tqdm(enumerate(dev_data_loader), total=dev_batch_count) as pbar:
            pbar.set_description('    Evaluating...')
            for i, (image, targets) in pbar:
                    images = Variable(images.to(device))
                    targets = Variable(targets.to(device))

                    with torch.no_grad():
                        output = model(images)
                        output = torch.sigmoid(output)

                    targets = targets.type_as(output)
                    gt = torch.cat((gt, targets), 0)
                    pred = torch.cat((pred, output.data), 0)
                    
        #Evaluate gt vs pred here!

        # Create output dir
        if not os.path.exists(opt.out_dir):
            os.makedirs(opt.out_dir)

        # Log results
        with open(opt.log, 'a+') as f:
            f.write('{}\n'.format(s))

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'hyp': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
            }
        }

        # Save last checkpoint
        torch.save(checkpoint, os.path.join(opt.out_dir, 'last.pkl'))

        # Save best checkpoint
        if bleu == best_bleu:
            torch.save(checkpoint, os.path.join(opt.out_dir, 'best.pkl'))

        # Save backup every 10 epochs (optional)
        if (epoch + 1) % save_after_x_epochs == 0:
            # Save our models
            print('!!! saving models at epoch: ' + str(epoch))
            torch.save(checkpoint, os.path.join(opt.out_dir, 'checkpoint-%d-%d.pkl' %(epoch+1, 1)))             

        # Delete checkpoint
        del checkpoint

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset path. Must contain training_set and eval_set subdirectories.')
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path for saving model weights')
    parser.add_argument('--weights', '-w', type=str, required=False, default='', help='weights to preload into model')
    parser.add_argument('--num-epochs', type=int, required=False, default=400, help='number of epochs')
    parser.add_argument('--learning-rate', '-lr', type=float, required=False, default=0.001, help='learning rate')
    parser.add_argument('--batch-size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--hidden_size', type=int, required=False, default=1024, help='dim of LSTM hidden layer')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--log', type=str, required=False, default='train.log', help='path to log file')
    parser.add_argument('--gpu-id', type=int, required=False, default=0, help='GPU ID to use')
    opt = parser.parse_args()
    main()
