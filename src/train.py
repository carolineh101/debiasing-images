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
    # pdb.set_trace()
    
    # Model Hyperparams
    hidden_size = opt.hidden_size
    learning_rate = opt.learning_rate
    save_after_x_epochs = 10

    # Determine device
    device = getDevice(opt.gpu_id)

    # Create data loaders

    data_loaders = load_celeba(splits=['train', 'valid'], batch_size=opt.batch_size, subset_percentage=opt.subset_percentage)
    train_data_loader = data_loaders['train']
    dev_data_loader = data_loaders['valid']

    # Create model
    model = BaselineModel(hidden_size)

    # Convert device
    model = model.to(device)

    # Create optimizer
    # criterion = nn.BCELoss()  # For multi-label classification
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = learning_rate)

    start_epoch = 0
    best_acc = 0.0
    save_best = False

    train_batch_count = len(train_data_loader)
    dev_batch_count = len(dev_data_loader)

    #Load if resume
    checkpoint = None
    if opt.weights != '':
        checkpoint = torch.load(opt.weights, map_location=device)
        model.load_state_dict(checkpoint['model'])    
        if opt.resume:
            if checkpoint['epoch']:
                start_epoch = checkpoint['epoch'] + 1
            if checkpoint['best_acc']:
                best_acc = checkpoint['best_acc']
            if checkpoint['optimizer']:
                optimizer.load_state_dict(checkpoint['optimizer']) 

    # Train loop
    # pdb.set_trace()
    for epoch in range(start_epoch, opt.num_epochs):

        # Set model to train mode
        model.train()

        # Initialize meters
        mean_accuracy = AverageMeter()
        mean_equality_gap_0 = AverageMeter()
        mean_equality_gap_1 = AverageMeter()
        mean_parity_gap = AverageMeter()

        with tqdm(enumerate(train_data_loader), total=train_batch_count) as pbar: # progress bar
            for i, (images, targets, genders) in pbar:

                # Shape: torch.Size([batch_size, 3, crop_size, crop_size])
                images = Variable(images.to(device))
                
                # Shape: torch.Size([batch_size, 39])
                targets = Variable(targets.to(device))

                # Shape: torch.Size([batch_size])
                genders = Variable(genders.to(device))

                # Zero out buffers
                # model.zero_grad() # either model or optimizer.zero_grad() is fine
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                targets = targets.type_as(outputs)
                genders = genders.type_as(outputs)

                # CrossEntropyLoss is expecting:
                # Input:  (N, C) where C = number of classes
                loss = criterion(outputs, targets)

                # Calculate accuracy
                train_acc = calculateAccuracy(outputs, targets)

                # Calculate fairness metrics
                train_equality_gap_0, train_equality_gap_1 = calculateEqualityGap(outputs, targets, genders)
                train_parity_gap = calculateParityGap(outputs, targets, genders)

                # Update averages
                mean_accuracy.update(train_acc, images.size(0))
                mean_equality_gap_0.update(train_equality_gap_0, images.size(0))
                mean_equality_gap_1.update(train_equality_gap_1, images.size(0))
                mean_parity_gap.update(train_parity_gap, images.size(0))

                # Backward pass
                loss.backward()
                optimizer.step()

                s_train = ('%10s Loss: %.4f, Accuracy: %.4f, Equality Gap 0: %.4f, Equality Gap 1: %.4f, Parity Gap: %.4f') % ('%g/%g' % (epoch, opt.num_epochs - 1), loss.item(), mean_accuracy.avg, mean_equality_gap_0.avg, mean_equality_gap_1.avg, mean_parity_gap.avg)
                pbar.set_description(s_train)

        # end batch ------------------------------------------------------------------------------------------------

        # Evaluate
        # pdb.set_trace()
        model.eval()

        # Initialize meters
        mean_accuracy = AverageMeter()
        mean_equality_gap = AverageMeter()
        mean_parity_gap = AverageMeter()

        with tqdm(enumerate(dev_data_loader), total=dev_batch_count) as pbar:
            for i, (images, targets, genders) in pbar:
                images = Variable(images.to(device))
                targets = Variable(targets.to(device))
                genders = Variable(genders.to(device))

                # gt = torch.cat((gt, targets), 0)

                with torch.no_grad():
                    # Forward pass
                    outputs = model(images)
                    targets = targets.type_as(outputs)
                    genders = genders.type_as(outputs)

                    # Calculate accuracy
                    eval_acc = calculateAccuracy(outputs, targets)

                    # Calculate fairness metrics
                    eval_equality_gap_0, eval_equality_gap_1 = calculateEqualityGap(outputs, targets, genders)
                    eval_parity_gap = calculateParityGap(outputs, targets, genders)

                    # Update averages
                    mean_accuracy.update(eval_acc, images.size(0))
                    mean_equality_gap_0.update(eval_equality_gap_0, images.size(0))
                    mean_equality_gap_1.update(eval_equality_gap_1, images.size(0))
                    mean_parity_gap.update(eval_parity_gap, images.size(0))

                    s_eval = ('%10s Accuracy: %.4f, Equality Gap 0: %.4f, Equality Gap 1: %.4f, Parity Gap: %.4f') % ('%g/%g' % (epoch, opt.num_epochs - 1), mean_accuracy.avg, mean_equality_gap_0.avg, mean_equality_gap_1.avg, mean_parity_gap.avg)
                    pbar.set_description(s_eval)

                # pred = torch.cat((pred, output.data), 0)

        # Create output dir
        if not os.path.exists(opt.out_dir):
            os.makedirs(opt.out_dir)

        # Log results
        with open(opt.log, 'a+') as f:
            f.write('{}\n'.format(s_train))
            f.write('{}\n'.format(s_eval))

        # Check against best accuracy
        mean_eval_acc = mean_accuracy.avg.cpu().item()
        if mean_eval_acc > best_acc:
            best_acc = mean_eval_acc
            save_best = True

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'hyp': {
                'hidden_size': hidden_size,
            }
        }

        # Save last checkpoint
        torch.save(checkpoint, os.path.join(opt.out_dir, 'last.pkl'))

        # Save best checkpoint
        if save_best:
            torch.save(checkpoint, os.path.join(opt.out_dir, 'best.pkl'))
            save_best = False

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
    # parser.add_argument('--dataset', type=str, required=True, help='dataset path. Must contain training_set and eval_set subdirectories.')
    parser.add_argument('--subset-percentage', type=float, required=False, default=1.0, help='Fraction of the dataset to use')
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path for saving model weights')
    parser.add_argument('--weights', '-w', type=str, required=False, default='', help='weights to preload into model')
    parser.add_argument('--num-epochs', type=int, required=False, default=400, help='number of epochs')
    parser.add_argument('--learning-rate', '-lr', type=float, required=False, default=0.001, help='learning rate')
    parser.add_argument('--batch-size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--hidden-size', type=int, required=False, default=1024, help='dim of hidden layer')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--log', type=str, required=False, default='train.log', help='path to log file')
    parser.add_argument('--gpu-id', type=int, required=False, default=0, help='GPU ID to use')
    opt = parser.parse_args()
    main()
