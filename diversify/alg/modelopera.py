# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from network import act_network


def get_fea(args):
    net = act_network.ActNetwork(args.dataset)
    return net


def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            if usedpredict == 'p':
                p = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
def predict_proba(network, x):
    network.eval()
    with torch.no_grad():
        logits = network.predict(x.cuda().float())
        probs = torch.nn.functional.softmax(logits, dim=1)
    network.train()
    return probs

    return correct / total
