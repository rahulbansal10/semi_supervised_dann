import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    label_criterion = nn.CrossEntropyLoss().cuda()
    loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
    ind_1_sorted = torch.argsort(loss_1.data).cpu().numpy()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
    ind_2_sorted = torch.argsort(loss_2.data).cpu().numpy()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    if(num_remember==0):
        num_remember = 1
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    # loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    # loss_2_update = F.cross_entropy(y_2[ind_2_update], t[ind_2_update])
    loss_1_update = label_criterion(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = label_criterion(y_2[ind_1_update], t[ind_1_update])

    # return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
    return (loss_1_update), (loss_2_update), pure_ratio_1, pure_ratio_2


def loss_coteaching2(y_1, y_2, t, d1, d2, domain_label, forget_rate, ind, noise_or_not):
    label_criterion = nn.CrossEntropyLoss().cuda()
    domain_criterion = nn.BCELoss().cuda()

    source_d1, target_d1 = d1.chunk(2,0)
    source_d2, target_d2 = d2.chunk(2,0)
    source_domain, target_domain = domain_label.chunk(2,0)

    loss_1 = F.cross_entropy(y_1, t, reduction = 'none')
    ind_1_sorted = torch.argsort(loss_1.data).cpu().numpy()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction = 'none')
    ind_2_sorted = torch.argsort(loss_2.data).cpu().numpy()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    if(num_remember==0):
        num_remember = 1
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    # loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    # loss_2_update = F.cross_entropy(y_2[ind_2_update], t[ind_2_update])
    loss_1_update = label_criterion(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = label_criterion(y_2[ind_1_update], t[ind_1_update])

    ld1 = domain_criterion(source_d1[ind_2_update], source_domain[ind_2_update]) + domain_criterion(target_d1, target_domain)
    ld2 = domain_criterion(source_d2[ind_1_update], source_domain[ind_1_update]) + domain_criterion(target_d2, target_domain)

    # return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
    return (loss_1_update), (loss_2_update), ld1, ld2, pure_ratio_1, pure_ratio_2