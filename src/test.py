import argparse
import numpy as np
import os
import pdb
import torch
import torchvision
from torch.autograd import Variable
from tqdm import tqdm

from dataset import load_celeba
from model import BaselineModel, OurModel
from utils import *

def main():
    # pdb.set_trace()

    # Determine device
    device = getDevice(opt.gpu_id)
    num_classes = 39

    # Create data loaders
    data_loaders = load_celeba(splits=['test'], batch_size=opt.batch_size, subset_percentage=opt.subset_percentage)
    test_data_loader = data_loaders['test']

    # Load checkpoint
    checkpoint = torch.load(os.path.join(opt.weights_dir, opt.out_dir, opt.weights), map_location=device)
    baseline = checkpoint['baseline']
    hidden_size = checkpoint['hyp']['hidden_size']

    # Create model
    if baseline:
        model = BaselineModel(hidden_size)
    else:
        model = OurModel(hidden_size)

    # Convert device
    model = model.to(device)

    test_batch_count = len(test_data_loader)

    # Load model
    model.load_state_dict(checkpoint['model'])    

    # Evaluate
    model.eval()

    # Initialize meters, confusion matrices, and metrics
    mean_accuracy = AverageMeter()
    attr_accuracy = AverageMeter((1, num_classes), device=device)
    cm_m = None
    cm_f = None
    attr_equality_gap_0 = None
    attr_equality_gap_1 = None
    attr_parity_gap = None

    with tqdm(enumerate(test_data_loader), total=test_batch_count) as pbar:
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

                s_test = ('Accuracy: %.4f') % (mean_accuracy.avg)

                # Calculate fairness metrics on final batch
                if i == test_batch_count - 1:
                    avg_equality_gap_0, avg_equality_gap_1, attr_equality_gap_0, attr_equality_gap_1 = \
                        calculateEqualityGap(cm_m, cm_f)
                    avg_parity_gap, attr_parity_gap = calculateParityGap(cm_m, cm_f)
                    s_test += (', Equality Gap 0: %.4f, Equality Gap 1: %.4f, Parity Gap: %.4f') % (avg_equality_gap_0, avg_equality_gap_1, avg_parity_gap)

                pbar.set_description(s_test)


        # Log results
        log_dir = os.path.join(opt.log_dir, opt.out_dir)
        with open(os.path.join(log_dir, opt.log), 'a+') as f:
            f.write('{}\n'.format(s_test))
        save_attr_metrics(attr_accuracy.avg, attr_equality_gap_0, attr_equality_gap_1, attr_parity_gap,
                          os.path.join(log_dir, opt.attr_metrics))

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True, help='dataset path. Must contain training_set and eval_set subdirectories.')
    parser.add_argument('--subset-percentage', type=float, required=False, default=1.0, help='Fraction of the dataset to use')
    parser.add_argument('--out-dir', type=str, required=True, help='subdirectory for logs and weights')
    parser.add_argument('--weights-dir', type=str, required=False, default='checkpoints', help='directory for weights')
    parser.add_argument('--weights', '-w', type=str, required=False, default='best.pkl', help='weights to preload into model')
    parser.add_argument('--batch-size', type=int, required=False, default=16, help='batch size')
    parser.add_argument('--log-dir', type=str, required=False, default='logs', help='directory for logs')
    parser.add_argument('--log', type=str, required=False, default='test.log', help='path to log file')
    parser.add_argument('--attr-metrics', type=str, required=False, default='test_attr', help='filename (to be prepended to \'.csv\') recording per-attribute metrics')
    parser.add_argument('--gpu-id', type=int, required=False, default=0, help='GPU ID to use')
    opt = parser.parse_args()
    main()
