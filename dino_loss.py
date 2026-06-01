import cv2
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision import transforms
from .dinov2 import vit_base



# build based on https://github.com/Hao840/manifold-distillation/blob/main/losses.py.
class L_manifold:
    def __init__(self, loss_type='CE'):
        self.loss_type = loss_type
        print('L_manifold initialized with cross entropy as loss type.')

    def __call__(self, f_F, f_dino):
        '''
            f_F/f_dino: b,c,h,w, features from original backbone and DINOv2.
            Calculate intra-image, inter-image, and random-indexed patches.
        '''
        # print(f_F, f_dino)
        
        # Align resolution.
        bsz, C_f, h, w = f_F.shape
        bsz, L, C_dino = f_dino.shape
        h_dino = int(L ** 0.5)
        f_dino = f_dino.permute(0,2,1).reshape(bsz, C_dino, h_dino, h_dino)
        f_dino = F.interpolate(f_dino, size=f_F.shape[2:], mode='bicubic')

        # convert to B,L,C.
        f_F = f_F.view(bsz, C_f, -1).permute(0,2,1)
        f_dino = f_dino.view(bsz, C_dino, -1).permute(0,2,1)

        # normalize along channel axis.
        
        f_F = F.normalize(f_F, dim=-1)
        f_dino = F.normalize(f_dino, dim=-1)
        # print(f_F.shape, f_dino.shape, f_F[0,0,:])

        # manifold loss among different patches (intra-sample).
        M_f_F = f_F.bmm(f_F.transpose(-1, -2))
        M_f_dino = f_dino.bmm(f_dino.transpose(-1, -2))

        if self.loss_type == 'l2':
            M_diff = M_f_F - M_f_dino
            loss_attn = (M_diff * M_diff).mean()
        elif self.loss_type == 'CE':
            eps = 1e-5
            M_f_dino = F.softmax(M_f_dino, dim=-1)
            M_f_F = F.softmax(M_f_F, dim=-1)

            loss_attn_batch = -(torch.log(M_f_F+eps) * M_f_dino).mean(-1) - (torch.log(M_f_dino+eps) * M_f_F).mean(-1)
            loss_attn = loss_attn_batch.mean()
        else:
            raise NotImplementedError

        return loss_attn


if __name__ == '__main__':

    from dinov2 import vit_base
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    dino_preprocess = torchvision.transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    dino_model = vit_base(
        patch_size=14,
        img_size=518,  # 14*37.
        init_values=1.0,
        block_chunks=0)  # block_chunks: (int) split block sequence into block_chunks units for FSDP wrap.
    with torch.no_grad():
        pretrained_weight = torch.load('/mnt/netdisk/liyf/DAI-Net/models/dinov2_vitb14_pretrain.pth')
        dino_model.load_state_dict(pretrained_weight)
        dino_model.eval()
        dino_extract = dino_model.cuda()

    loss = L_manifold()
    from PIL import Image
    import numpy as np
    x = Image.open('/mnt/netdisk/wangwenjing/Datasets/CODaN/train/Bicycle/n02835271_10064.JPEG')
    x = torch.tensor(np.array(x).transpose(2,0,1)).cuda().unsqueeze(0).repeat(2,1,1,1) / 255.
    x = x.repeat(32, 1, 1, 1)

    with torch.no_grad():
        h_, w_ = x.shape[-2] // 14 * 14, x.shape[-1] // 14 * 14
        x_for_dino = dino_preprocess(F.interpolate(x, size=(h_, w_)))
        dino_feat = dino_extract.get_intermediate_layers(x_for_dino)[0]

    f_F = torch.rand(64,64,56,56).cuda()
    l = loss(f_F, dino_feat.detach())

    print(l)
