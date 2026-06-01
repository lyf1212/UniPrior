import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


# Color Invariant Convolutions based on CIConv (ICCV'21) and QuadPrior (CVPR'24).
eps = 1e-7

# This code is based on https://github.com/Attila94/CIConv/blob/main/method/ciconv2d.py.

def gaussian_basis_filters(scale, gpu, k=3):
    std = torch.pow(2,scale)

    # Define the basis vector for the current scale
    filtersize = torch.ceil(k*std+0.5)
    x = torch.arange(start=-filtersize.item(), end=filtersize.item()+1)
    if gpu is not None: x = x.to(gpu); std = std.to(gpu)
    x = torch.meshgrid([x,x])

    # Calculate Gaussian filter base
    # Only exponent part of Gaussian function since it is normalized anyway
    g = torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    g = g / (torch.sum(g) + eps) # Normalize

    # Gaussian derivative dg/dx filter base
    dgdx = -x[0]/((std**3*2*math.pi)+eps)*torch.exp(-(x[0]/std)**2/2)*torch.exp(-(x[1]/std)**2/2)
    dgdx = dgdx / (torch.sum(torch.abs(dgdx)) + eps)  # Normalize

    # Gaussian derivative dg/dy filter base
    dgdy = -x[1]/(std**3*2*math.pi)*torch.exp(-(x[1]/std)**2/2)*torch.exp(-(x[0]/std)**2/2)
    dgdy = dgdy / (torch.sum(torch.abs(dgdy)) + eps)  # Normalize

    # Stack and expand dim
    basis_filter = torch.stack([g,dgdx,dgdy], dim=0)[:,None,:,:]

    return basis_filter

def convolve_gaussian_filters(batch, scale):
    E, El, Ell = torch.split(batch, 1, dim=1)
    E_out, El_out, Ell_out = [], [], []

    for s in range(len(scale)):
        # Convolve with Gaussian filters
        w = gaussian_basis_filters(scale=scale[s:s+1], gpu=batch.device).to(dtype=batch.dtype)  # KCHW

        # the padding here works as "same" for odd kernel sizes
        E_out.append(F.conv2d(input=E[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        El_out.append(F.conv2d(input=El[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))
        Ell_out.append(F.conv2d(input=Ell[s:s+1,:,:,:], weight=w, padding=int(w.shape[2]/2)))

    return torch.cat(E_out), torch.cat(El_out), torch.cat(Ell_out)

# =================================
# == Color invariant definitions ==
# =================================

def hat_H(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    # avoid nan.
    near_zeros = Ell.abs() < 1e-10
    Ell = Ell * (near_zeros.logical_not())
    Ell = Ell + (near_zeros * eps)
    return torch.atan(El/Ell)

def hat_S(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    return (El**2 + Ell**2) / (E**2 +eps) 

def hat_Ww(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    Wx = Ex / (E+eps)
    Wy = Ey / (E+eps)
    return Wx**2 + Wy**2

def hat_Wlw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    Wlx = Elx / (E+eps)
    Wly = Ely / (E+eps)
    return Wlx**2 + Wly**2

def hat_Wllw2(E,Ex,Ey,El,Elx,Ely,Ell,Ellx,Elly):
    Wllx = Ellx / (E+eps)
    Wlly = Elly / (E+eps)
    return Wllx**2 + Wlly**2

# =================================
# == Color invariant convolution ==
# =================================

class PriorConv2d(nn.Module):
    def __init__(self):

        super(PriorConv2d, self).__init__()

        # Constants
        self.gcm = torch.nn.Parameter(torch.tensor([[0.06,0.63,0.27],[0.3,0.04,-0.35],[0.34,-0.6,0.17]]))
            
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            torch.nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, batch):
        # Make sure scale does not explode: clamp to max abs value of 2.5
        # self.scale.data = torch.clamp(self.scale.data, min=-2.5, max=2.5)

        with torch.no_grad():
            if batch.shape[1] != 3:
                print('WARNING: current batch.shape[1] != 3')
                return torch.zeros((batch.shape[0], 6, batch.shape[2], batch.shape[3]), device=batch.device)
            # compute twice for max_RGB and max_RGB_: 
            # in case for equal RGB value. If there RGB are not, max_RGB==min_RGB_.
            max_RGB = torch.argmax(batch, dim=1)
            min_RGB = torch.argmin(batch, dim=1)

            batch_ = torch.flip(batch, dims=(1,))

            max_RGB_ = 2 - torch.argmax(batch_, dim=1)
            min_RGB_ = 2 - torch.argmin(batch_, dim=1)

            # print(torch.allclose(min_RGB_, max_RGB))
            RGB_order = torch.zeros(batch.shape, device=batch.device, dtype=batch.dtype)
            RGB_order = RGB_order.scatter_(1, max_RGB.unsqueeze(1), 0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, max_RGB_.unsqueeze(1), 0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, min_RGB.unsqueeze(1), -0.5, reduce='add')
            RGB_order = RGB_order.scatter_(1, min_RGB_.unsqueeze(1), -0.5, reduce='add')
            # print(RGB_order.shape)
        if torch.isnan(batch).any():
            print('input is Nan.')
        scale = torch.mean(self.conv(batch), dim=(1,2,3))  # TODO: after 80k (bsz=4), it becomes NaN.
        if torch.isnan(scale).any():
            print('scale is Nan.')
        scale = torch.clamp(scale, min=-2.5, max=2.5) 

        # Measure E, El, Ell by Gaussian color model
        in_shape = batch.shape  # bchw
        batch = batch.view((in_shape[:2]+(-1,)))  # flatten image
        batch = torch.matmul(self.gcm.to(batch.device, dtype=batch.dtype), batch)  # estimate E,El,Ell
        batch = batch.view((in_shape[0],)+(3,)+in_shape[2:])  # reshape to original image size

        E_out, El_out, Ell_out = convolve_gaussian_filters(batch.float(), scale.float())

        
        # print("W1")
        E, Ex, Ey = torch.split(E_out,1,dim=1)
        El = torch.split(El_out,1,dim=1)[0]
        Ell = torch.split(Ell_out,1,dim=1)[0]

        H = hat_H(E,Ex,Ey,El,None,None,Ell,None,None)
        S = torch.log(hat_S(E,Ex,Ey,El,None,None,Ell,None,None)+eps)
        Ww = torch.atan(hat_Ww(E,Ex,Ey,El,None,None,Ell,None,None))

        features = torch.cat([H, S, RGB_order, Ww], dim=1)
        return features.to(dtype=batch.dtype)



class IIM_kubelka(nn.Module):
    '''
        Illumination Invariant Module with only Kubelka-Munk prior.

        return priors which have the same spatial resolutions with input x.
    '''
    def __init__(self, pretrain=True):
        super().__init__()
        
        self.iim_block = PriorConv2d()
        self.norm = nn.InstanceNorm2d(6, affine=True)

        # for first time setup.
        # state_dict = torch.load('/mnt/netdisk/liyf/QuadPrior-main/checkpoints/COCO-final.ckpt', map_location='cpu')
        # prior_conv_state = {}
        # for name in state_dict.keys():
            # if 'prior' in name:
                # prior_conv_state[name.replace('_forward_module.control_model.prior_conv.', '')] = state_dict[name]
        # print(prior_conv_state.keys())

        if pretrain:
            from .path_register import PRIOR_PATH
            prior_conv_state = torch.load(PRIOR_PATH)
            self.iim_block.load_state_dict(prior_conv_state)

        # for first time setup.
        # torch.save(self.iim_block2.state_dict(), './prior_conv.pth')

    def forward(self, x):
        
        feat_ii = self.iim_block(x)  # B,6,h,w.
        feat_ii = self.norm(feat_ii)

        return feat_ii
