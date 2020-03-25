import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import numpy as np

class SegNet(nn.Module):

    def __init__(self, opt, input_dim):

        super(SegNet, self).__init__()

        self.opt = opt

        self.conv1 = nn.Conv2d(input_dim, opt.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(opt.nChannel)
        
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(opt.nConv-1):
            self.conv2.append( nn.Conv2d(opt.nChannel, opt.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(opt.nChannel) )
        
        # 1x1 convolution
        self.conv3 = nn.Conv2d(opt.nChannel, opt.nClass, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(opt.nClass)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.opt.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        feats = x.clone()
        x = self.conv3(x)
        out = self.bn3(x)
        
        return feats, out

class DiscriminativeLoss(_Loss):
    
    def __init__(self, delta_var=0.5, delta_dist=15,
                 norm=2, alpha=1.0, beta=0.01, gamma=0.001, size_average=True):
        super(DiscriminativeLoss, self).__init__(size_average)

        self.device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
        self.delta_var = torch.Tensor([delta_var]).to(self.device)
        self.delta_dist = torch.Tensor([delta_dist]).to(self.device)
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        assert self.norm in [1, 2]

    def forward(self, input, pred_clusters, n_clusters):
        return self._discriminative_loss(input, pred_clusters, n_clusters)

    def _discriminative_loss(self, input, pred_clusters, n_clusters):
        
        means = torch.zeros((n_clusters, input.shape[1])).to(self.device)
        cluster_nums = np.unique(pred_clusters.data.cpu().numpy())

        for i in range(len(cluster_nums)):
            embeds = input[pred_clusters==cluster_nums[i]]
            means[i] = torch.sum(embeds, dim=0) / len(embeds)

        # Variance
        l_var = 0
        for i in range(len(cluster_nums)):
            embeds = input[pred_clusters==cluster_nums[i]]
            var = torch.sum(torch.clamp(torch.norm((embeds - means[i]), self.norm, 1) - self.delta_var, min=0)**2)
            l_var += 1/(len(embeds)+1e-5) * var
        l_var /= n_clusters

        # Distance
        l_dist = 0
        n_features = means.shape[1]
        
        means_a = means.permute(1,0).unsqueeze(2).expand(n_features, n_clusters, n_clusters)
        means_b = means_a.permute(0, 2, 1)
        diff = means_a - means_b
        diff = torch.sum(diff**2+1e-3, dim=0)**0.5

        margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters).to(self.device))
        margin = Variable(margin).to(self.device)
        
        c_dist = torch.sum(torch.sum(torch.clamp(margin - diff, min=0)))
        l_dist += c_dist / (2 * n_clusters * (n_clusters - 1))

        # Normalize
        l_reg = 1 / n_clusters * torch.sum(torch.sum(means**2, dim=1)**0.5)

        l_all = self.alpha*l_var + self.beta*l_dist + self.gamma*l_reg
        
        return l_all
        
        
        


    
    



