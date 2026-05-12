import os
import sys
import datetime
import time
import itertools
import warnings
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np

from tqdm import tqdm

from data_processing import get_data_feats

from models import *
from utils import *
from metrics import *
from custom_dataset import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SAGL')

parser.add_argument('--dataset', type=str, default='pets',
                    choices=['pets', 'kitti', 'flowers', 'caltech101', 'eurosat', 'sun397', 'food101', 'imagenet',
                              'dtd','stl10', 'aircraft', 'sst'], help='Dataset name')
parser.add_argument('--model_names', type=str, default=['siglip', 'dinov3'], nargs='+', help='At least two pretrained models',
                            choices=['clipvitB32', 'clipvitB16', 'clipvitL14', 'dinov2', 'siglip', 'dinov3', 'convnext_v2'])
parser.add_argument('--root_dir', type=str, default="data", help='Root dir stored feature data')
parser.add_argument('--load_model', default=False, help='Testing if True or training.')
parser.add_argument('--save_model', default=True, help='Saving the model after training.')
parser.add_argument('--model_name', default='pets_model.pth', help='the model^s name.')

parser.add_argument('--seed', type=int, default=42, help='Initializing random seed')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for task encoder')
parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=600)

parser.add_argument('--alpha', type=float, default=5.0, help='Cluster diversity regularization')
parser.add_argument('--beta', type=float, default=1.0, help='Cross-view consistency regularization')
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--testing_shuffle', default=False, help='The testing dataset is shuffled if True')
parser.add_argument('--factor_divided', default=True, help='The dynamic sparsity gate factor obtained by conditional division if True')

parser.add_argument('--gpu', default=0, type=int, help='GPU device idx')


args = parser.parse_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@torch.no_grad()
def test_seq(model, test_loader, labels):
    model.eval()

    all_soft_labels = []
    for batch_feats, batch_labels in test_loader:
        soft_labels, _, _ = model(batch_feats)
        predicted_labels = soft_labels.argmax(dim=1).detach().cpu().numpy()
        all_soft_labels.extend(predicted_labels)
    acc, nmi, pur, ari = calculate_metrics(np.array(all_soft_labels), labels)

    return acc, nmi, pur, ari

@torch.no_grad()
def test(model, test_loader):
    model.eval()

    all_soft_labels = []
    all_labels = []
    for batch_feats, batch_labels in test_loader:
        soft_labels, _, _ = model(batch_feats)
        predicted_labels = soft_labels.argmax(dim=1).detach().cpu().numpy()
        all_soft_labels.extend(predicted_labels)
        all_labels.extend(batch_labels)
    acc, nmi, pur, ari = calculate_metrics(np.array(all_soft_labels), np.array(all_labels))

    return acc, nmi, pur, ari

def train_with_multiple_models(model, optimizer, train_loader):
    model.train()

    total_loss = 0.0
    for batch_feats in train_loader:
        optimizer.zero_grad()
        predicted_error = 0.0
        labels, probs_set, _ = model(batch_feats)
        pseudo_labels = labels.argmax(dim=1)

        for predicted_labels in probs_set:
            predicted_error += F.nll_loss(predicted_labels, pseudo_labels.detach())

        label_loss = 0.0
        entr_reg = 0.0
        log_probs_set = [torch.log(p) for p in probs_set]

        for i, j in itertools.combinations(range(len(probs_set)), 2):
            x, y = probs_set[i], probs_set[j]
            log_x = log_probs_set[i]
            label_loss += -torch.mean(torch.sum(log_x * y, dim=1))
            entr_reg += forward_prob(x, y)

        loss = predicted_error + args.alpha * entr_reg + args.beta * label_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def train(model, optimizer, train_loader):
    model.train()

    eps = 1e-8
    total_loss = 0.0
    for batch_feats in train_loader:
        optimizer.zero_grad()
        predicted_error = 0.0
        labels, probs_set, _ = model(batch_feats)
        pseudo_labels = labels.argmax(dim=1)

        probs_set = [p.clamp(min=eps, max=1.0) for p in probs_set]
        for predicted_labels in probs_set:
            predicted_error += F.nll_loss(predicted_labels, pseudo_labels.detach())

        label_loss = -torch.mean(torch.sum(torch.log(probs_set[0]) * probs_set[1], dim=1))

        entr_reg = forward_prob(probs_set[0], probs_set[1])

        loss = predicted_error + args.alpha * entr_reg + args.beta * label_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def set_default_config():
    if args.dataset == 'pets':  # v1
        args.lr = 1e-3
        args.batch_size = 100
        args.alpha = 5.0
        args.beta = 0.5
        args.dropout = 0.1

    elif args.dataset == 'flowers':
        args.lr = 5e-4
        args.batch_size = 500
        args.alpha = 5.0
        args.beta = 1.0
        args.dropout = 0.0

    elif args.dataset == 'kitti':
        args.lr = 1e-3
        args.batch_size = 100
        args.alpha = 10.0
        args.beta = 0.5
        args.dropout = 0.1

    elif args.dataset == 'caltech101':
        args.lr = 1e-3
        args.batch_size = 200
        args.alpha = 5.0
        args.beta = 1.0
        args.dropout = 0.2

    elif args.dataset == 'eurosat':
        args.lr = 1e-3
        args.batch_size = 500
        args.alpha = 10.0
        args.beta = 0.1
        args.dropout = 0.0


    elif args.dataset == 'sun397':
        args.lr = 1e-3
        args.batch_size = 5000
        # args.batch_size = 200
        args.alpha = 10.0
        args.beta = 0.2
        args.dropout = 0.1

    elif args.dataset == 'food101':
        args.lr = 5e-4
        args.batch_size = 5000
        # args.batch_size = 200
        args.alpha = 20.0
        args.beta = 0.1
        args.dropout = 0.1

    elif args.dataset == 'imagenet':
        args.lr = 5e-4
        args.batch_size = 10000
        args.alpha = 5.0
        args.beta = 0.2
        args.dropout = 0.3

    else:
        raise NotImplementedError

if __name__ == '__main__':

    # set_default_config()
    # model_parent_dir = './models/models_%s_%d/' % (args.dataset, args.gpu)
    #
    # num_models = len(args.model_names)
    # assert num_models >= 2, 'The number of models is at least two.'
    #
    # set_seed(args.seed)
    # print(f'Loading dataset: {args.dataset}')
    # train_feat_sets, test_feat_sets, train_labels, test_labels = get_data_feats(args.root_dir, args.dataset,
    #                                                                           args.model_names, device)
    # num_class = len(np.unique(train_labels))
    # num_train_sample = len(train_labels)
    # num_test_sample = len(test_labels)
    # print(f"The number of the training samples: {num_train_sample}")
    # print(f"The number of the test samples: {num_test_sample}")
    # print(f"The number of classes: {num_class}")
    # print(f'Features of {args.model_names}: ' + ' '.join(str(train_feats.shape) for train_feats in train_feat_sets))
    #
    # train_dataset = CustomSimplifiedDataset(train_feat_sets)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #
    # test_dataset = CustomDataset(test_feat_sets, torch.from_numpy(test_labels))
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    #
    # if not args.load_model:
    #     # torch.cuda.empty_cache()  # for ImageNet with btach_size >= 10000
    #
    #     start_time = time.time()
    #     model_dir, record_time = create_dir_model(model_parent_dir)
    #     input_dims = [train_feats.shape[1] for train_feats in train_feat_sets]
    #
    #     model = SAGL(input_dims, num_class, args.dropout, args.factor_divided)
    #     # if args.dataset == "imagenet":
    #     #     model = torch.nn.DataParallel(model, device_ids=[0, 1]) # for ImageNet
    #     model.to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     pbar = tqdm(range(1, args.epochs+1), desc=f'Training', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=120)
    #
    #     for epoch in pbar:
    #         total_loss = 0.0
    #         if num_models == 2:
    #             total_loss = train(model, optimizer, train_loader)
    #         else:
    #             total_loss = train_with_multiple_models(model, optimizer, train_loader)
    #
    #         # evaluation
    #         if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
    #             if args.testing_shuffle:
    #                 acc, nmi, pur, ari = test(model, test_loader)
    #             else:
    #                 acc, nmi, pur, ari = test_seq(model, test_loader, test_labels)
    #             time_cost = time.time() - start_time
    #             pbar.set_postfix(loss=f'{total_loss:.3f}', test_acc=f'{acc:.4f}')
    #             with open('SAGL_%s_%s_result_%s.txt' % (args.dataset, '_'.join(args.model_names), args.gpu), 'a+') as f:
    #                 f.write('{} \t {:.4f} \t {} \t {} \t {:.1f} \t {:.1f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.2f} \n'.format(
    #                     record_time, args.lr, args.batch_size, epoch, args.alpha, args.beta, args.dropout, acc, nmi, pur, ari, time_cost))
    #                 f.flush()
    #
    #             if args.save_model and (epoch == 1 or epoch >= 60):
    #                 if not os.path.exists(model_dir):
    #                     os.mkdir(model_dir)
    #                 torch.save(model.state_dict(), f'{model_dir}/{args.dataset}_{epoch}_model.pth')
    #             # pbar.set_description(f'Training loss {float(total_loss):.3f}, cluster acc {acc:.4f}')
    #     print(f'Finished')
    #
    # else:
    #     input_dims = [train_feats.shape[1] for train_feats in train_feat_sets]
    #     model = SAGL(input_dims, num_class, args.dropout)
    #     # if args.dataset == "imagenet":
    #     #     model = torch.nn.DataParallel(model, device_ids=[0, 1]) # for ImageNet
    #     model.to(device)
    #     assert args.model_name.strip() != '', 'The name of the model is empty.'
    #     state_dict = torch.load(f'models/{args.model_name}')
    #     model.load_state_dict(state_dict)
    #     acc, nmi, pur, ari = test(model, test_loader, test_labels)
    #     print(f'Final testing cluster acc: {acc:.4f}, nmi: {nmi:.4f}, pur: {pur:.4f}, ari: {ari:.4f}')

    #Uncomment the following section to search for appropriate parameters.
    batch_sizes = np.array([10000, 8000, 5000, 2000], dtype=np.int32)  # for large-scale datasets
    # batch_sizes = np.array([2000, 1000, 500, 200], dtype=np.int32) # for medium-scale datasets
    # batch_sizes = np.array([100, 200, 500, 1000], dtype=np.int32) # for small-scale datasets
    learning_rates = np.array([5e-4, 1e-3], dtype=np.float32)
    alphas = np.array([5.0, 10.0, 20.0], dtype=np.float32)
    betas = np.array([1.0, 0.5, 0.2, 0.1, 0.05], dtype=np.float32)
    dropouts = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    # dropouts = np.array([0.2, 0.3, 0.4], dtype=np.float32) for large-scale datasets

    set_seed(args.seed)
    print(f'Loading dataset: {args.dataset}')
    train_feat_sets, test_feat_sets, train_labels, test_labels = get_data_feats(args.root_dir, args.dataset,
                                                                              args.model_names, device)

    model_parent_dir = './models/models_%s_%d/' % (args.dataset, args.gpu)
    model_dir, record_time = create_dir_model(model_parent_dir)
    input_dims = [train_feats.shape[1] for train_feats in train_feat_sets]
    num_class = len(np.unique(train_labels))
    num_train_sample = len(train_labels)
    num_test_sample = len(test_labels)

    num_models = len(args.model_names)
    assert num_models >= 2, 'The number of models is at least two.'

    print(f"The number of the training samples: {num_train_sample}")
    print(f"The number of the test samples: {num_test_sample}")
    print(f"The number of classes: {num_class}")
    print(f'Features of {args.model_names}: ' + ' '.join(str(train_feats.shape) for train_feats in train_feat_sets))

    for batch_idx in range(batch_sizes.shape[0]):
        args.batch_size = int(batch_sizes[batch_idx])

        train_dataset = CustomSimplifiedDataset(train_feat_sets)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = CustomDataset(test_feat_sets, torch.from_numpy(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.testing_shuffle)

        for lr_idx in range(learning_rates.shape[0]):
            args.lr = learning_rates[lr_idx]
            for alpha_idx in range(alphas.shape[0]):
                args.alpha = alphas[alpha_idx]
                for beta_idx in range(betas.shape[0]):
                    args.beta = betas[beta_idx]

                    for dropout_idx in range(dropouts.shape[0]):
                        args.dropout = dropouts[dropout_idx]

                        # torch.cuda.empty_cache()  # for ImageNet
                        start_time = time.time()
                        model = SAGL(input_dims, num_class, args.dropout, args.factor_divided)
                        # if args.dataset == "imagenet":
                        #     model = torch.nn.DataParallel(model, device_ids=[0, 1]) # for ImageNet
                        model.to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                        pbar = tqdm(range(1, args.epochs + 1), desc=f'Training',
                                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', ncols=120)
                        for epoch in pbar:
                            if num_models == 2:
                                total_loss = train(model, optimizer, train_loader)
                            else:
                                total_loss = train_with_multiple_models(model, optimizer, train_loader)
                            # evaluation
                            if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
                                if args.testing_shuffle:
                                    acc, nmi, pur, ari = test(model, test_loader)
                                else:
                                    acc, nmi, pur, ari = test_seq(model, test_loader, test_labels)
                                time_cost = time.time() - start_time
                                pbar.set_postfix(loss=f'{total_loss:.3f}', test_acc=f'{acc:.4f}')
                                with open('SAGL_%s_%s_result_%s.txt' % (args.dataset, '_'.join(args.model_names), args.gpu), 'a+') as f:
                                    f.write(
                                        '{} \t {:.4f} \t {} \t {} \t {:.1f} \t {:.1f} \t {:.1f} \t {:.2f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.5f} \n'.format(
                                            record_time, args.lr, args.batch_size, epoch, args.alpha, args.beta,
                                            args.dropout, acc * 100, nmi, pur, ari, time_cost * 0.001))
                                    f.flush()

                                if args.save_model and (epoch == 1 or epoch >= 100):
                                    if not os.path.exists(model_dir):
                                        os.mkdir(model_dir)
                                    torch.save(model.state_dict(), f'{model_dir}/{args.dataset}_{epoch}_model.pth')
                                # pbar.set_description(f'Training loss {float(total_loss):.3f}, cluster acc {cluster_acc:.4f}')
                        print(f'Finished')

