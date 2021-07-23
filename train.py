# from __future__ import print_function
from utils import *
from datasets import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import PreResNet
import math
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import models
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Analyzing hidden states when training with mixup on latent representations for LNL.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training, default: 128')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing, default: 100')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train, default: 10')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate, default: 0.1')
    parser.add_argument('--dataset', type=str, default='Spiral', choices=[
                        'spiral', 'mnist'], help='dataset to train on, default: spiral')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default: 0.9')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA support')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed, set it to go to determinist mode. We used 1 for the paper, default: None')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status, default: 100')
    parser.add_argument('--noise-level', type=float, default=80.0,
                        help='percentage of noise added to the data (values from 0. to 100.), default: 80.')
    parser.add_argument('--experiment-name', type=str, default='runs',
                        help='name of the experiment for the output files storage, default: runs')
    parser.add_argument('--alpha', type=float, default=32,
                        help='alpha parameter for the mixup distribution, default: 32')
    parser.add_argument('--M', nargs='+', type=int, default=[30, 60],
                        help="Milestones for the LR sheduler, default 30 60")
    parser.add_argument('--Mixup', type=str, default='None', choices=['None', 'Static', 'Hidden', 'Dynamic', 'Hidden-Dynamic'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Hidden' (adapted for hidden states from Static) \
                                'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None \
                                'Hidden-Dynamic' (adapted for hidden states from Dynamic)")
    parser.add_argument('--reg-term', type=float, default=0.,
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--tb-dir', type=str, default=None,
                        help="Tensorboard output directory, default: no Tensorboard support.")

    args = parser.parse_args()

    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))

    if args.seed:
        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(args.seed)  # CPU seed
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)  # GPU seed

        random.seed(args.seed)  # python seed for image transformation

    model = None

    if args.dataset == 'Spiral':
        trainset, trainset_track, testset = get_spiral_datasets(args.root_dir)
        model = models.SpiralModel().to(device)
    # elif args.dataset == 'MNIST':
    #     train_loader, train_loader_track, test_loader = get_mnist_loaders(args.root_dir,
    #                                                                       args.batch_size, args.test_batch_size)
    #     model = models.MNISTModel().to(device)
    else:
        raise NotImplementedError
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    train_loader_track = torch.utils.data.DataLoader(
        trainset_track, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)    

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)

    labels = get_data_dataset_2(train_loader_track)  # it should be "cloning"
    # it changes the labels in the train loader directly
    noisy_labels = add_noise_dataset_w(train_loader, args.noise_level)
    noisy_labels_track = add_noise_dataset_w(
        train_loader_track, args.noise_level)

    # path where experiments are saved
    exp_path = os.path.join(
        './', 'noise_models', '{0}_{1}'.format(args.dataset, args.experiment_name), str(args.noise_level))

    # tensorboard

    if args.tb_dir:
        tb = SummaryWriter(os.path.join(
            "./", "tensorboard", args.tb_dir, "{0}-{1}".format(args.experiment_name, str(args.noise_level))))

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    bmm_model = bmm_model_maxLoss = bmm_model_minLoss = cont = k = 0

    # the +1 is because the conditions are defined as ">" or "<" not ">="
    bootstrap_ep_std = milestones[0] + 5 + 1 
    bootstrap_ep_mixup = milestones[0] + 5 + 1


    for epoch in range(1, args.epochs + 1):
        # train
        scheduler.step()

        ### Standard CE training (without mixup) ###
        if args.Mixup == "None":
            print('\t##### Doing standard training with cross-entropy loss #####')
            loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(
                args, model, device, train_loader, optimizer, epoch)

        ### Mixup ###
        # just analyze with M-DYR-H and latent space variations of it
        if args.Mixup == "Static":
            alpha = args.alpha
            if epoch < bootstrap_ep_mixup:
                print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(
                    bootstrap_ep_mixup - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(
                    args, model, device, train_loader, optimizer, epoch, 32)

            else:
                print("\t##### Doing HARD BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(
                    bootstrap_ep_mixup))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch,
                                                                                alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)                                                                                     alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)

        ### Hidden State Mixup ###
        if args.Mixup == "Hidden":
            alpha = args.alpha
            if epoch < bootstrap_ep_mixup:
                print('\t##### Doing HIDDEN mixup for {0} epochs #####'.format(
                    bootstrap_ep_mixup - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(
                    args, model, device, train_loader, optimizer, epoch, 32, hidden_mixup=True)

            else:
                print("\t##### Doing HARD BETA bootstrapping and HIDDEN mixup from the epoch {0} #####".format(
                    bootstrap_ep_mixup))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch,
                                                                                 alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes, hidden_mixup=True)


        # tensorboard
        if tb:
            tb.add_scalar("train/loss", loss_per_epoch[-1], epoch)
            tb.add_scalar("train/accuracy", acc_train_per_epoch_i[-1], epoch)
            tb.flush()

        # Training tracking loss
        epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
            track_training_loss(args, model, device, train_loader_track,
                                epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)

        # test
        loss_per_epoch, acc_val_per_epoch_i = test_cleaning(
            args, model, device, test_loader)

        # tensorboard
        if tb:
            tb.add_scalar("val/loss", loss_per_epoch[-1], epoch)
            tb.add_scalar("val/accuracy", acc_val_per_epoch_i[-1], epoch)
            tb.flush()

        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            torch.save(model.state_dict(), os.path.join(
                exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(
                exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]

                if cont > 0:
                    try:
                        os.remove(os.path.join(
                            exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
                torch.save(model.state_dict(), os.path.join(
                    exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(
                    exp_path, 'opt_' + snapBest + '.pth'))

        cont += 1

        if epoch == args.epochs:
            snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestValLoss_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            torch.save(model.state_dict(), os.path.join(
                exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(
                exp_path, 'opt_' + snapLast + '.pth'))


if __name__ == '__main__':
    main()
