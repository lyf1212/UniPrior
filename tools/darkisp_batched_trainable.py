import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import random
import torchvision
import torch.nn.functional as F


class DarkISP(nn.Module):

    '''
        Learnable Degradation Pipeline based on ISP.
    '''
    def __init__(self, device='cpu'):
        super().__init__()

        # learnable parameters.
        self.darkness_ratio = nn.Parameter(torch.tensor([0.1721]))
        self.gamma_ratio = nn.Parameter(torch.tensor([2.2]))  # 1.6.
        self.rgb_range = nn.Parameter(torch.tensor([0.76]))  # 0.76.
        self.red_range = nn.Parameter(torch.tensor([2.05]))  # 2.05.
        self.blue_range = nn.Parameter(torch.tensor([1.76]))  # 1.76.
        self.quantisation = nn.Parameter(torch.tensor([12.0]))
        self.log_min_shot_noise = nn.Parameter(torch.tensor([np.log(0.0001)], dtype=torch.float32))
        self.log_max_shot_noise = nn.Parameter(torch.tensor([np.log(0.012)], dtype=torch.float32))
        self.read_noise_std = nn.Parameter(torch.tensor([0.26]))

        # camera color matrix
        self.xyz2cams = [[[1.0234, -0.2969, -0.2266],
                        [-0.5625, 1.6328, -0.0469],
                        [-0.0703, 0.2188, 0.6406]],
                    [[0.4913, -0.0541, -0.0202],
                        [-0.613, 1.3513, 0.2906],
                        [-0.1564, 0.2151, 0.7183]],
                    [[0.838, -0.263, -0.0639],
                        [-0.2887, 1.0725, 0.2496],
                        [-0.0627, 0.1427, 0.5438]],
                    [[0.6596, -0.2079, -0.0562],
                        [-0.4782, 1.3016, 0.1933],
                        [-0.097, 0.1581, 0.5181]]]
        self.rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]]
        self.device = device

        self.noisy_iters = 1


    def apply_ccm(self, image, ccm):
        '''
        The function of apply CCM matrix
        '''
        shape = image.shape
        image = image.view(shape[0], -1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        return image.view(shape) 
    

    def random_noise_levels(self, log_min_shot_noise, log_max_shot_noise, read_noise_std, bsz):
        """Generates random shot and read noise from a log-log linear distribution."""

        interval = log_max_shot_noise - log_min_shot_noise
        log_shot_noise = torch.rand(bsz, device=self.device, dtype=torch.float32) / interval + log_min_shot_noise
        shot_noise = torch.exp(log_shot_noise)

        log_read_noise = 2.18 * log_shot_noise + 0.12 + torch.randn(bsz, device=self.device, dtype=torch.float32) * 0.26  # read_noise_std is set to a const here. 
        read_noise = torch.exp(log_read_noise)

        return shot_noise.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), read_noise.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


    def forward(self, img, use_random_darken=False):
        '''
            Modified based on https://github.com/cuiziteng/ICCV_MAET.
            (1) reverse ISP (RGB-->RAW) 
            (2) low light corruption  
            (3) forward ISP (RAW-->RGB)
            input:
            img (Tensor): normal-lit, (B,C,H,W), RGB.

            return:
            img_deg (Tensor): synthesized low-lit images, (B,C,H,W), RGB.
        '''
        # noise parameters and quantization step

        '''
        (1)unprocess part(RGB2RAW): 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains
        '''
        img1 = img.permute(0, 2, 3, 1)  # (B, C, H, W) -- (B, H, W, C)
        # inverse tone mapping with deterministic mapping.
        img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
        # inverse gamma correction.
        epsilon = torch.tensor([1e-6], device=self.device)
        img2 = torch.max(img1, epsilon) ** self.gamma_ratio
        # sRGB2cRGB.
        xyz2cam = random.choice(self.xyz2cams)
        # print(xyz2cam)
        rgb2cam = np.matmul(xyz2cam, self.rgb2xyz)
        rgb2cam = torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1)).to(torch.float32).to(self.device)
        img3 = self.apply_ccm(img2, rgb2cam)

        # inverse WB
        gains1 = torch.stack([1.0 / self.red_range, torch.tensor([1.0], device=self.device), 1.0 / self.blue_range], dim=-1) * self.rgb_range
        gains1 = gains1.unsqueeze(0).unsqueeze(0)  # (1,1,1,3).

        img4 = img3 * gains1

        # color disorder !!!
        # img3_gray = torch.mean(img3, dim=-1, keepdim=True)
        # inflection = 0.9
        # zero = torch.zeros_like(img3_gray, device=self.device)
        # mask = (torch.max(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
        # safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)
        # img4 = torch.clamp(img3*safe_gains, min=0.0, max=1.0)

        '''
        (2)low light corruption part: 5.darkness, 6.shot and read noise 
        '''
        # linear darken.
        if use_random_darken:
            darken_ratio = torch.ones_like(img4) * self.darkness_ratio
            random_weight = torch.rand(size=(img4.shape[0], 1, img4.shape[1]//32, img4.shape[2]//32), device=img4.device) * 0.3 + 0.7
            random_weight = F.interpolate(random_weight, size=img4.shape[1:3], mode='bicubic')
            # print(random_weight.shape, img4.shape[1:3])
            darken_ratio = darken_ratio * random_weight.permute(0,2,3,1)
            img5 = img4 * darken_ratio
        else:
            img5 = img4 * self.darkness_ratio
        
        # add shot and read noise (bsz, constants).
        for _ in range(self.noisy_iters):
            shot_noise, read_noise = self.random_noise_levels(self.log_min_shot_noise, self.log_max_shot_noise, self.read_noise_std, bsz=img5.shape[0])
            var = img5 * shot_noise + read_noise
            var = torch.max(var, epsilon)
            noise = torch.randn_like(var) * torch.sqrt(var)
            img5 = img5 + noise  # no inplace operation for proper back-propogation.

        img6 = img5

        '''
        (3)ISP part(RAW2RGB): 7.quantisation  8.white balance 9.cRGB2sRGB 10.gamma correction
        '''
        # quantisation noise: uniform distribution
        quan_noise = (torch.rand(img6.size(), device=self.device) * 2 - 1) * (1 / (255 * self.quantisation * 2))
        # print(quan_noise)
        # img7 = torch.clamp(img6 + quan_noise, min=0)
        img7 = img6 + quan_noise
        # white balance
        gains2 = torch.stack([self.red_range, torch.tensor([1.0], device=self.device), self.blue_range], dim=-1)
        gains2 = gains2.unsqueeze(0).unsqueeze(0)

        img8 = img7 * gains2
        # cRGB2sRGB
        cam2rgb = torch.inverse(rgb2cam)
        img9 = self.apply_ccm(img8, cam2rgb)
        # gamma correction
        img10 = torch.max(img9, epsilon) ** (1 / self.gamma_ratio)

        img_low = img10.permute(0, 3, 1, 2)  # (B, H, W, C) -- (B, C, H, W)
        self.degrade_para = {'dark': self.darkness_ratio.detach().cpu(), 
                         'gamma': self.gamma_ratio.detach().cpu(), 
                         'rgb_range': self.rgb_range.detach().cpu(),
                         'red_gain': self.red_range.detach().cpu(),
                         'blue_gain': self.blue_range.detach().cpu(),
                         'quantisation': self.quantisation.detach().cpu(),
                         'log_min_shot_noise': self.log_min_shot_noise.detach().cpu(),
                         'log_max_shot_noise': self.log_max_shot_noise.detach().cpu(),
                         'read_noise_std': self.read_noise_std.detach().cpu()}
        return img_low
    

if __name__ == '__main__':

    import cv2
    img = cv2.imread('/mnt/netdisk/liyf/Low_Light_Datasets/CODaN/train/Cat/000000000650.jpg')[..., ::-1] / 255.
    img_t = torch.tensor(img.transpose(2,0,1)).type(torch.float32)
    img_t = torch.stack([img_t, img_t])
    print(img_t.shape)
    degrade_pipe = DarkISP(device='cuda').cuda()
    print(list(degrade_pipe.parameters()))
    degrade_t = degrade_pipe(img_t.cuda())[0]
    print(degrade_pipe.degrade_para)
    cv2.imwrite('./darken.png', (degrade_t.detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)[..., ::-1])