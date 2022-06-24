import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import torch


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode = 'fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std = 1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1: ])).item()
        return x.view(-1, shape)

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize = 3, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, transpose = False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size = ksize, stride = stride, padding = padding, dilation = dilation, groups = groups,
            bias = bias)
            
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class Drift(nn.Module):
    def __init__(self, dim):
        super(Drift, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
    
    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class Diffusion(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Diffusion, self).__init__()
        self.norm1 = norm(dim_in)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = ConcatConv2d(dim_in, dim_out, 3, 1, 1)
        self.norm2 = norm(dim_in)
        self.conv2 = ConcatConv2d(dim_in, dim_out, 3, 1, 1)
        self.fc = nn.Sequential(norm(dim_out), 
                                nn.ReLU(inplace = True), 
                                nn.AdaptiveAvgPool2d((1, 1)), 
                                Flatten(), 
                                nn.Linear(dim_out, 1), 
                                nn.Sigmoid())

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)                       
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.fc(out)
        return out

class SDENetClassification(nn.Module):
    '''
    Input: MRI-Conductivities from CondNet
    Shape: (1, 249, 249, 176) with channels: 176
    '''
    def __init__(self, layer_depth, num_classes = 10, dim = 64):
        super(SDENetClassification, self).__init__()
        self.layer_depth = layer_depth 
        self.downsampling_layers = nn.Sequential(
                                            nn.Conv2d(1, dim, 3, 1),
                                            norm(dim),
                                            nn.ReLU(inplace = True),
                                            nn.Conv2d(dim, dim, 4, 2),
                                            norm(dim),
                                            nn.ReLU(inplace = True),
                                            nn.Conv2d(dim, dim, 4, 2))
        self.drift = Drift(dim)
        self.diffusion = Diffusion(dim, dim)
        self.fc_layers = nn.Sequential(
            norm(dim),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim, 10)
        )
        self.deltat = 6./self.layer_depth
        self.apply(init_params)
        self.sigma = 500
    
    def forward(self, x, training_diffusion = False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma * self.diffusion(t, out)
            diffusion_term = torch.unsqueeze(diffusion_term, 2)
            diffusion_term = torch.unsqueeze(diffusion_term, 3)
            for i in range(self.layer_depth):
                t = 6 * (float(i))/self.layer_depth
                out = out + self.drift(t, out) * self.deltat + diffusion_term * math.sqrt(self.deltat) * torch.randn_like(out).to(x)
            final_out = self.fc_layers(out)
        else:
            t = 0
            final_out = self.diffusion(t, out.detach())
        return final_out

class Drift(nn.Module):
    '''
    Aims to control the system to achieve a good predictive accuracy
    '''
    def __init__(self):
        super(Drift, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, t, x):
        return self.relu(self.fc(x))

class Diffusion(nn.Module):
    '''
    The diffusion net satisfies the following:
        [1] For regions in the training distributionm the variance of the Brownian motion
            should be small (low diffusion). The system is dominated by the drift term in
            this area and the output variance should be small.
            
        [2] For regions outside the training distribution, the variance of the Brownian 
            motion should be large and the system is chaotic (high diffusion). In this case,
            the variance of the outputs for multiple evaluations should be large.
    '''
    def __init__(self):
        super(Diffusion, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.fc1  = nn.Linear(1, 100)
        self.fc2  = nn.Linear(100, 1)
    
    def forward(self, t, x):            
        out = self.relu(self.fc1(x))    
        out = self.fc2(out)             
        out = torch.sigmoid(out)        
        return out

class SDENetRegression(nn.Module):
    def __init__(self, layer_depth = 6):
        super(SDENetRegression, self).__init__()
        self.layer_depth = layer_depth
        self.downsampling_layers = nn.Linear(1, 1)           
        self.fc_layers = nn.Sequential(
            nn.ReLU(inplace = True),
            nn.Linear(1, 1))

        self.drift = Drift()
        self.diffusion = Diffusion()
        self.deltat = 4.0 / self.layer_depth
        self.sigma = 0.5

    def forward(self, x, training_diffusion = False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma * self.diffusion(t, out)
            drift_term = self.drift(t, out)
            for i in range(self.layer_depth):
                t = 4 * (float(i)) / self.layer_depth
                wiener_process = torch.randn_like(out, requires_grad= True).to(x)
                out = (out  
                            + (drift_term * self.deltat) 
                            + (diffusion_term * math.sqrt(self.deltat) * wiener_process))
            
            final_out = self.fc_layers(out)
            mean = final_out[:, 0]
            sigma = F.softplus(final_out[:]) + 1e-3
            return mean, sigma
        else:
            t = 0 
            final_out = self.diffusion(t, out.detach())
            return final_out