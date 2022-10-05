import argparse
import time
import json
import numpy as np
import math
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence

from model import Model

def seed_everything(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',   type=str,    default="model/")
    parser.add_argument('--batch_size',  type=int,    default=128)
    parser.add_argument('--epochs',      type=int,    default=200) 
    parser.add_argument('--runs',        type=int,    default=5)
    parser.add_argument('--domain',      type=str,    default="laptop")
    parser.add_argument('--data_dir',    type=str,    default="data/prep_data/")
    parser.add_argument('--valid',       type=int,    default=150)
    parser.add_argument('--lr',          type=float,  default=0.0001) 
    parser.add_argument('--dropout',     type=float,  default=0.55)
    parser.add_argument("--seed",        type=int,    default=1337) 
    parser.add_argument("--cuda",        type=str,    default='1')

    args = parser.parse_args()
    return args

def generate_batch(X, y, batch_size=128, use_cuda=True):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_cnt = np.sum(X[offset:offset+batch_size] != 0, axis=1)
        batch_idx = np.argsort(batch_X_cnt)[::-1]
        batch_X_cnt = batch_X_cnt[batch_idx]

        batch_X_mask = X[offset:offset+batch_size] != 0
        batch_X_mask = batch_X_mask[batch_idx]
        batch_X_mask.astype(np.uint8)
        batch_X_mask = Variable(torch.from_numpy(batch_X_mask).long())

        batch_X = X[offset:offset+batch_size]
        batch_X = batch_X[batch_idx]
        batch_X = Variable(torch.from_numpy(batch_X).long())

        batch_y = y[offset:offset+batch_size]
        batch_y = batch_y[batch_idx]
        batch_y = Variable(torch.from_numpy(batch_y).long())

        batch_y = pack_padded_sequence(batch_y, batch_X_cnt, batch_first=True)

        if use_cuda:
            batch_X_mask = batch_X_mask.cuda()
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()

        yield batch_X_cnt, batch_X_mask, batch_X, batch_y

def calculate_loss(model, X, y, batch_size, use_cuda):
    model.eval()
    losses = []
    for batch in generate_batch(X, y, batch_size, use_cuda):
        loss = model(batch[0], batch[1], batch[2], batch[3])
        losses.append(loss.item())
    model.train()
    return np.array(losses).mean()


def train(data, epochs, config, use_cuda, optimizer, model, model_file):
    best_loss = float("inf")
    for epoch in range(epochs):
        for batch in generate_batch(data['train_X'], data['train_y'], config['batch_size'], use_cuda):
            loss = model(batch[0], batch[1], batch[2], batch[3])
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(model.parameters(), config['max_norm'])
            optimizer.step()
        loss = calculate_loss(model, data['train_X'], data['train_y'], config['batch_size'], use_cuda)
        loss = calculate_loss(model, data['train_X'], data['train_y'], config['batch_size'], use_cuda)
        if min(loss, best_loss) == loss:
            best_loss = loss
            torch.save(model, model_file)
            shuffle_idx = np.random.permutation(len(data['train_X']))
        data['train_X'] = data['train_X'][shuffle_idx]
        data['train_y'] = data['train_y'][shuffle_idx]

def run(runs, general_embedding, domain_embedding, data, config, use_cuda, epochs, model_dir):
    for r in range(runs):
        print(r)
        model = Model(general_embedding, 
                      domain_embedding, 
                      config['num_labels'],
                      config['dropout_prob'],
                      config['nums_filter'],
                      config['kernel_sizes'],
                      config['nums_words'])
        if use_cuda:
            model.cuda()

        # Adam optimizer
        optimizer = Adam(model.parameters(), lr=config['lr'])

        # train
        model_file = model_dir + str(r)
        train(data, epochs, config, use_cuda, optimizer, model, model_file)



if __name__ == "__main__":
    # get args
    args = get_args()

    # cuda
    use_cuda = int(args.cuda) >= 0

    # fix the seed for reproducibility
    seed_everything(args.seed, use_cuda)

    # load embeddings
    general_embedding = np.load(args.data_dir + "gen.vec.npy")
    domain_embedding = np.load(args.data_dir + args.domain + "_emb.vec.npy")

    # load ae data
    ae_data = np.load(args.data_dir + args.domain + ".npz")
    ae_data_size = len(ae_data['train_X'])
    train_size = ae_data_size - args.valid

    # split ae data into train and validation datasets
    train_X = ae_data['train_X'][:train_size]
    train_y = ae_data['train_y'][:train_size]
    valid_X = ae_data['train_X'][train_size:]
    valid_y = ae_data['train_y'][train_size:]

    data = {'train_X': train_X, 'train_y': train_y, 'valid_X': valid_X, 'valid_y': valid_y}
    config = {'num_labels': 3, 'dropout_prob': 0.5, 'lr': args.lr, 'batch_size': args.batch_size,
              'max_norm': 1., 'nums_filter': [128, 256], 'kernel_sizes': [5, 3, 5], 'nums_words': [2, 1]}

    run(args.runs, general_embedding, domain_embedding, data, config, use_cuda, args.epochs, args.model_dir + args.domain)



