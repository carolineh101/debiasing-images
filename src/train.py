import argparse
import numpy as np
import os
import pdb
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import random

from tqdm import tqdm

from dataset import load_celeba
from model import BaselineModel, OurModel
from utils import *

def main():
    
    # Model Hyperparams
    random.seed(opt.random_seed)
    baseline = opt.baseline
    hidden_size = opt.hidden_size
    lambd = opt.lambd
    learning_rate = opt.learning_rate
    adv_learning_rate = opt.adv_learning_rate
    save_after_x_epochs = 10
    num_classes = 39

    # Determine device
    device = getDevice(opt.gpu_id)

    # Create data loaders
    data_loaders = load_celeba(splits=['train', 'valid'], batch_size=opt.batch_size, subset_percentage=opt.subset_percentage, \
         protected_percentage = opt.protected_percentage)
    train_data_loader = data_loaders['train']
    dev_data_loader = data_loaders['valid']

    # Load checkpoint
    checkpoint = None
    if opt.weights != '':
        checkpoint = torch.load(opt.weights, map_location=device)
        baseline = checkpoint['baseline']
        hidden_size = checkpoint['hyp']['hidden_size']

    # Create model
    if baseline:
        model = BaselineModel(hidden_size)
    else:
        model = OurModel(hidden_size)

    # Convert device
    model = model.to(device)

    # Loss criterion
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    if not baseline:
        adversarial_criterion = nn.BCEWithLogitsLoss()

    # Create optimizers
    primary_optimizer_params = list(model.encoder.parameters()) + list(model.classifier.parameters())
    primary_optimizer = torch.optim.Adam(primary_optimizer_params, lr = learning_rate)
    if not baseline:
        adversarial_optimizer_params = list(model.adv_head.parameters())
        adversarial_optimizer = torch.optim.Adam(adversarial_optimizer_params, lr = adv_learning_rate)

    start_epoch = 0
    best_acc = 0.0
    save_best = False

    train_batch_count = len(train_data_loader)
    dev_batch_count = len(dev_data_loader)

    if checkpoint is not None:
        # Load model weights
        model.load_state_dict(checkpoint['model']) 

        # Load metadata to resume training   
        if opt.resume:
            if checkpoint['epoch']:
                start_epoch = checkpoint['epoch'] + 1
            if checkpoint['best_acc']:
                best_acc = checkpoint['best_acc']
            if checkpoint['hyp']['lambd']:
                lambd = checkpoint['lambd']
            if checkpoint['optimizers']['primary']:
                primary_optimizer.load_state_dict(checkpoint['optimizers']['primary'])
            if checkpoint['optimizers']['adversarial']:
                adversarial_optimizer.load_state_dict(checkpoint['optimizers']['adversarial'])

    # Train loop
    # pdb.set_trace()
    for epoch in range(start_epoch, opt.num_epochs):

        # Set model to train mode
        model.train()

        # Initialize meters and confusion matrices
        mean_accuracy = AverageMeter(device=device)
        cm_m = None
        cm_f = None

        with tqdm(enumerate(train_data_loader), total=train_batch_count) as pbar: # progress bar
            for i, (images, targets, genders, protected_labels) in pbar:

                # Shape: torch.Size([batch_size, 3, crop_size, crop_size])
                images = Variable(images.to(device))
                
                # Shape: torch.Size([batch_size, 39])
                targets = Variable(targets.to(device))

                # Shape: torch.Size([batch_size])
                genders = Variable(genders.to(device))

                # Shape: torch.Size([batch_size])
                protected_labels = Variable(protected_labels.type(torch.BoolTensor).to(device))

                # Forward pass
                if baseline:
                    outputs, (a, a_detached) = model(images)
                else:
                    outputs, (a, a_detached) = model(images, protected_labels)
                targets = targets.type_as(outputs)
                genders = genders.type_as(outputs)

                # Zero out buffers
                # model.zero_grad() # either model or optimizer.zero_grad() is fine
                primary_optimizer.zero_grad()

                # CrossEntropyLoss is expecting:
                # Input:  (N, C) where C = number of classes
                classification_loss = criterion(outputs, targets)

                if baseline:
                    loss = classification_loss
                else:
                    adversarial_loss = adversarial_criterion(a, genders[protected_labels])
                    loss = classification_loss - lambd * adversarial_loss

                    # Backward pass (Primary)
                    loss.backward()
                    primary_optimizer.step()

                    # Zero out buffers
                    adversarial_optimizer.zero_grad()

                    # Calculate loss for adversarial head
                    adversarial_loss = adversarial_criterion(a_detached, genders[protected_labels])

                    # Backward pass (Adversarial)
                    adversarial_loss.backward()
                    adversarial_optimizer.step()


                # Convert genders: (batch_size, 1) -> (batch_size,)
                genders = genders.view(-1).bool()

                # Calculate accuracy
                train_acc, _ = calculateAccuracy(outputs, targets)

                # Calculate confusion matrices
                batch_cm_m, batch_cm_f = calculateGenderConfusionMatrices(outputs, targets, genders)
                if cm_m is None and cm_f is None:
                    cm_m = batch_cm_m
                    cm_f = batch_cm_f
                else:
                    cm_m = list(cm_m)
                    cm_f = list(cm_f)
                    for j in range(len(cm_m)):
                        cm_m[j] += batch_cm_m[j]
                        cm_f[j] += batch_cm_f[j]
                    cm_m = tuple(cm_m)
                    cm_f = tuple(cm_f)

                # Update averages
                mean_accuracy.update(train_acc, images.size(0))

                if baseline:
                    s_train = ('%10s Loss: %.4f, Accuracy: %.4f') % ('%g/%g' % (epoch, opt.num_epochs - 1), loss.item(), mean_accuracy.avg)
                else:
                    s_train = ('%10s Classification Loss: %.4f, Adversarial Loss: %.4f, Total Loss: %.4f, Accuracy: %.4f') % ('%g/%g' % (epoch, opt.num_epochs - 1), classification_loss.item(), adversarial_loss.item(), loss.item(), mean_accuracy.avg)

                # Calculate fairness metrics on final batch
                if i == train_batch_count - 1:
                    avg_equality_gap_0, avg_equality_gap_1, _, _ = calculateEqualityGap(cm_m, cm_f)
                    avg_parity_gap, _ = calculateParityGap(cm_m, cm_f)
                    s_train += (', Equality Gap 0: %.4f, Equality Gap 1: %.4f, Parity Gap: %.4f') % (avg_equality_gap_0, avg_equality_gap_1, avg_parity_gap)

                pbar.set_description(s_train)

        # end batch ------------------------------------------------------------------------------------------------

        # Evaluate
        # pdb.set_trace()
        model.eval()

        # Initialize meters, confusion matrices, and metrics
        mean_accuracy = AverageMeter()
        attr_accuracy = AverageMeter((1, num_classes), device=device)
        cm_m = None
        cm_f = None
        attr_equality_gap_0 = None
        attr_equality_gap_1 = None
        attr_parity_gap = None

        with tqdm(enumerate(dev_data_loader), total=dev_batch_count) as pbar:
            for i, (images, targets, genders, protected_labels) in pbar:
                images = Variable(images.to(device))
                targets = Variable(targets.to(device))
                genders = Variable(genders.to(device))

                with torch.no_grad():
                    # Forward pass
                    outputs = model.sample(images)
                    targets = targets.type_as(outputs)

                    # Convert genders: (batch_size, 1) -> (batch_size,)
                    genders = genders.type_as(outputs).view(-1).bool()

                    # Calculate accuracy
                    eval_acc, eval_attr_acc = calculateAccuracy(outputs, targets)

                    # Calculate confusion matrices
                    batch_cm_m, batch_cm_f = calculateGenderConfusionMatrices(outputs, targets, genders)
                    if cm_m is None and cm_f is None:
                        cm_m = batch_cm_m
                        cm_f = batch_cm_f
                    else:
                        cm_m = list(cm_m)
                        cm_f = list(cm_f)
                        for j in range(len(cm_m)):
                            cm_m[j] += batch_cm_m[j]
                            cm_f[j] += batch_cm_f[j]
                        cm_m = tuple(cm_m)
                        cm_f = tuple(cm_f)

                    # Update averages
                    mean_accuracy.update(eval_acc, images.size(0))
                    attr_accuracy.update(eval_attr_acc, images.size(0))

                    s_eval = ('%10s Accuracy: %.4f') % ('%g/%g' % (epoch, opt.num_epochs - 1), mean_accuracy.avg)

                    # Calculate fairness metrics on final batch
                    if i == dev_batch_count - 1:
                        avg_equality_gap_0, avg_equality_gap_1, attr_equality_gap_0, attr_equality_gap_1 = \
                            calculateEqualityGap(cm_m, cm_f)
                        avg_parity_gap, attr_parity_gap = calculateParityGap(cm_m, cm_f)
                        s_eval += (', Equality Gap 0: %.4f, Equality Gap 1: %.4f, Parity Gap: %.4f') % (avg_equality_gap_0, avg_equality_gap_1, avg_parity_gap)

                    pbar.set_description(s_eval)


        # Create output dir
        if not os.path.exists(opt.out_dir):
            os.makedirs(opt.out_dir)

        # Log results
        with open(opt.log, 'a+') as f:
            f.write('{}\n'.format(s_train))
            f.write('{}\n'.format(s_eval))
        save_attr_metrics(attr_accuracy.avg, attr_equality_gap_0, attr_equality_gap_1, attr_parity_gap,
                          opt.attr_metrics + '_' + str(epoch))

        # Check against best accuracy
        mean_eval_acc = mean_accuracy.avg.cpu().item()
        if mean_eval_acc > best_acc:
            best_acc = mean_eval_acc
            save_best = True

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizers': {
                'primary': primary_optimizer.state_dict(),
                'adversarial': adversarial_optimizer.state_dict() if not baseline else None,
            },
            'best_acc': best_acc,
            'baseline': baseline,
            'hyp': {
                'hidden_size': hidden_size,
                'lambd': lambd
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
    parser.add_argument('--protected-percentage', type=float, required=False, default=1.0, help='Fraction of dataset with protected class label')
    parser.add_argument('--out-dir', '-o', type=str, required=True, help='output path for saving model weights')
    parser.add_argument('--weights', '-w', type=str, required=False, default='', help='weights to preload into model')
    parser.add_argument('--num-epochs', type=int, required=False, default=10, help='number of epochs')
    parser.add_argument('--learning-rate', '-lr', type=float, required=False, default=0.0001, help='learning rate')
    parser.add_argument('--adv-learning-rate', '-alr', type=float, required=False, default=0.001, help='adversarial learning rate')
    parser.add_argument('--batch-size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--hidden-size', type=int, required=False, default=1024, help='dim of hidden layer')
    parser.add_argument('--lambd', type=float, required=False, default=0.1, help='adversarial weight hyperparameter, lambda')
    parser.add_argument('--baseline', action='store_true', help='train baseline model (without adversarial head')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--log', type=str, required=False, default='train.log', help='path to log file')
    parser.add_argument('--attr-metrics', type=str, required=False, default='train_attr', help='filename (to be prepended to \'_{epoch}.csv\') recording per-attribute metrics')
    parser.add_argument('--gpu-id', type=int, required=False, default=0, help='GPU ID to use')
    parser.add_argument('--random-seed', type=int, required=False, default=1, help='random seed')
    opt = parser.parse_args()
    main()
