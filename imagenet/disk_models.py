
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.distributions import Categorical

class ImageNet_Routing_B1(nn.Module):
    def __init__(self, n_labels=1000, ):
        print('---------------------------------- ImageNet B1 (teacher_logits) --')
        super(ImageNet_Routing_B1, self).__init__()

        self.n_labels = n_labels
        n_ft = 64

        self.routing = nn.Sequential(
                nn.Linear(n_labels, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
                nn.Linear(n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, 1), # bias=True),
                nn.Sigmoid(),
            )

    def forward(self, t_logits, t_ft=None ):
        gate = self.routing( t_logits )
        return gate  

class ImageNet_Routing_B2(nn.Module):
    def __init__(self, n_labels=1000, num_features=-1):
        print('---------------------------------- ImageNet B2 (teacher_ft) --')
        super(ImageNet_Routing_B2, self).__init__()

        self.n_labels = n_labels
        n_ft = 64

        self.routing = nn.Sequential(
                nn.Linear(num_features, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
                nn.Linear(n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, 1), # bias=True),
                nn.Sigmoid(),
            )

    def forward(self, t_logits, t_ft ):
        #gate = self.routing( t_logits )
        #print('t_ft ', t_ft.size())
        gate = self.routing( t_ft )
        return gate  

class ImageNet_Routing_B3(nn.Module):
    def __init__(self, n_labels=1000, num_features=-1):
        print('---------------------------------- ImageNet B3 (teacher_ft) --')
        super(ImageNet_Routing_B3, self).__init__()

        self.n_labels = n_labels
        n_ft = 256


        self.routing1 = nn.Sequential(
                nn.Linear(num_features, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, n_ft, bias=True),
                nn.BatchNorm1d( n_ft ),
                nn.ReLU(),
            )

        self.routing = nn.Sequential(
                nn.Linear(n_ft, n_labels, bias=True),
                nn.BatchNorm1d( n_labels ),
                nn.ReLU(),
                nn.Linear(n_labels, 1), # bias=True),
                nn.Sigmoid(),
            )

    def forward(self, t_logits, t_ft ):
        #gate = self.routing( t_logits )
        #print('t_ft ', t_ft.size())
        x = self.routing1( t_ft )
        x = F.normalize( x )
        gate = self.routing( x )
        return gate  




def get_gating_model( routing_name = 'ImageNet_Routing_B1', n_labels=1000, t_num_ft=-1 ):
    if routing_name=='ImageNet_Routing_B1':
        routingNet = ImageNet_Routing_B1( n_labels = n_labels )
    elif routing_name=='ImageNet_Routing_B2':
        routingNet = ImageNet_Routing_B2( n_labels = n_labels, num_features=t_num_ft )
    elif routing_name=='ImageNet_Routing_B3':
        routingNet = ImageNet_Routing_B3( n_labels = n_labels, num_features=t_num_ft )
    else:
        assert(1==2)
    return routingNet
