import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .saliency import calculate_saliency  # FFT-based traditional method for key point extraction.

class MaskContrast_Inter(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=32, T=3/36.5, patch_size=7):
        '''
            DINO provide a regularization of feature relationship.
            To narrow the gap between normal-lit and low-light, we close the distance of the same region between nl and ll.
            To enlarge the gap between different clusters in an image, we pull them away.
            Such regularization is employed in two images, a low-light image and a normal-lit image.
        '''
        super().__init__()
        from .contrasitive import ContrastiveMLPConv
        self.proj_head = ContrastiveMLPConv(in_channel=input_dim, out_channel=output_dim, bottle_channel=hidden_dim)
        
        # self.proj_head = nn.Sequential(
        #     nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
        #     nn.SiLU(),
        #     nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
        #     nn.SiLU(),
        #     nn.AdaptiveAvgPool2d(1),
        # )
        self.T = T
        self.patch_size = patch_size

    def patchify(self, x):
        _,c,h,w = x.shape
        x = x.reshape(-1, c, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size)
        x = x.permute(0,1,2,4,3,5).reshape(-1,c,self.patch_size,self.patch_size).contiguous()  # b*grid^2,c,patch,patch.
        return x
        

    def forward(self, x, x_aug, mask):
        '''
        mask: [b,grid,grid], 1 means positive anchor, 0 means negative sample.
        x: [b,c,h,w], a shallow feature which may be disrupted by degradation.
        x_aug: [b,c,h,w], normal version of image.
        '''
        mask = F.interpolate(mask.unsqueeze(1), size=(x.shape[2]//self.patch_size, x.shape[3]//self.patch_size))
        # torchvision.utils.save_image(mask, 'debug_resized_mask.png')
        # print(mask.shape)
        mask = mask.flatten()  # b*grid^2, .
        x, x_aug = map(lambda x: self.patchify(x), [x, x_aug])  # b*grid^2,c,patch,patch.
        # print('1', x.shape)
        # print('x', x)
        x_emb = self.proj_head(x).flatten(1)
        x_aug_emb = self.proj_head(x_aug).flatten(1)
        # print('x_emb', x_emb)
        # print('x_emb.shape', x_emb.shape)
        x_anchor = x_emb[mask!=0]  # pos_tokens, hidden_dim, remains mask==1 patches.
        x_pos = x_aug_emb[mask!=0]  # pos_tokens, hidden_dim, remains mask==1 patches.
        x_neg = x_emb[mask==0]  # neg_tokens, hidden_dim, remains mask==0 patches.
        

        x_anchor, x_pos, x_neg = map(lambda x: F.normalize(x, dim=-1), [x_anchor, x_pos, x_neg])
        # print(x_anchor.shape, x_neg.shape)
        l_pos = (x_anchor * x_pos).sum(-1, keepdim=True)  # inner product of positive samples, pull them together -> 1.
        # print('l_pos', l_pos.shape)  # pos_tk, 1.
        l_neg = x_anchor @ x_neg.T  # inner product of pos-neg, pull them away -> 0.
        # print('l_neg',l_neg.shape)  # pos_tk, neg_tk.

        logits = torch.cat([l_pos, l_neg], dim=1) / self.T  # pos_tk, 1+neg_tk, contract 1st column, pull the negs away.
        softmax_logits = F.softmax(logits, dim=-1)
        # print('softmax_logits', softmax_logits)
        
        # only one positive: when temperature==1, InfoNCE loss is equal to Cross-Entropy loss.
        loss = - torch.log(softmax_logits[:, 0]).mean()

        return loss


def generate_cluster_mask(dino_model, x: torch.Tensor, grid_size: int, cos_sim_tile_thres=0.6, offset_grid_size=2, sample_point_n=3, use_saliency=False):
    '''
        x: [b,c,h,w], h == 14 * grid_size, by default is 518 (grid_size=37).
    '''
    bsz = x.shape[0]
    assert grid_size * 14 == x.shape[-1], 'You messed up the shapes.'
    
    with torch.inference_mode():
        tokens = dino_model.get_intermediate_layers(x)[0]  # [b, grid_size^2, 768].
    tokens_n = F.normalize(tokens, dim=-1)
    dis = tokens_n @ tokens_n.transpose(1,2)  # [b, grid_size^2, grid_size^2].
    # print('dis=', dis.shape, dis)

    mask = torch.zeros_like(dis)
    # print(torch.quantile(dis, cos_sim_tile_thres, dim=-1, keepdim=True).shape)
    # print(torch.quantile(dis, cos_sim_tile_thres, dim=-1)[:,None,:].shape)

    # Caution: broadcast the scale p-percentile across dim=-1, make sure each item in dim=-1 can be compared with the p-percentile along its axis.
    # mask[dis > torch.quantile(dis, cos_sim_tile_thres, dim=-1)[:,None,:]] = 1  # [b, grid_size^2, grid_size^2].
    thres_ada = torch.quantile(dis, cos_sim_tile_thres, dim=-1, keepdim=True)  # [b, grid_size^2, 1].
    mask[dis > thres_ada] = 1  # [b, grid_size^2, grid_size^2].
    # print(mask.shape)
    # torchvision.utils.save_image(mask[0,21].reshape(37, 37), 'debug_mask.png')

    if use_saliency:
        # random select one key points based on saliency map for each item in a batch.
        saliency = calculate_saliency(x)  # b,h*w.
        key_points = torch.multinomial(saliency, 1)  # b,1.
        center_wh = torch.concat([key_points%grid_size, key_points//grid_size], dim=1).type(torch.int64)  # b,2.
    else:
        # select the key point which is correspondent to the max confidence (reflected by quantiled threshold).
        center_id = torch.argmax(thres_ada[..., 0], dim=1, keepdim=True)  # b,.
        # print(center_id)
        center_wh = torch.concat([center_id%grid_size, center_id//grid_size], dim=1).type(torch.int64)  # b,2.
    
    # print(center_wh.shape, center_wh)

    # random sample from the grid of center anchor point.
    # Do you know why the below value range [-4, 4] theoretically, but actually [-3, 3]? (when offset_size=3).
    # because of `.type(int)` quantization, we need to enlarge random value range.
    offset = (torch.rand((bsz, sample_point_n, 2), device=x.device) * (2*(offset_grid_size+1)) - offset_grid_size-1).type(torch.int64)
    # print(center_wh.device, offset.device)

    sample_points = center_wh[:,None,:] + offset
    sample_token_ids = sample_points[:,:,0] + sample_points[:,:,1] * grid_size  # Causion: which is width/height?.
    # Causion: avoid array out of index.
    sample_token_ids = torch.clamp(sample_token_ids, 0, grid_size*grid_size-1)

    batch_indices = torch.arange(bsz).unsqueeze(-1).expand(-1, sample_point_n)
    mask_cluster = mask[batch_indices, sample_token_ids]
    mask_cluster = mask_cluster.sum(1).clamp_(0, 1).reshape(bsz, grid_size, -1)
    # print(mask_cluster.shape)
    # torchvision.utils.save_image(mask_cluster[0], './debug_mask_func.png')


    return mask_cluster



if __name__ == '__main__':

    import numpy as np
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import torchvision

    from dinov2 import vit_base
    model = vit_base(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        block_chunks=0)  # block_chunks: (int) split block sequence into block_chunks units for FSDP wrap.
    pretrained_weight = torch.load('/mnt/netdisk/liyf/DAI-Net/models/dinov2_vitb14_pretrain.pth')
    model.load_state_dict(pretrained_weight)
    model.eval()
    model = model.cuda()
    print(f"patch size: {model.patch_size}")

    example_image = Image.open('/mnt/netdisk/liyf/Low_Light_Datasets/CODaN/train/Cat/000000000650.jpg')
    example_image = example_image.resize((518, 518))

    img_t = torch.tensor(np.array(example_image).transpose(2,0,1)/255)
    img_t = torch.stack([img_t, img_t, img_t]).float().cuda()
    
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    preprocess = torchvision.transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    img_t = preprocess(img_t)

    print(img_t.shape)

    y = generate_cluster_mask(dino_model=model, x=img_t, grid_size=37).cuda()
    print(y.shape)
    y = F.interpolate(y.unsqueeze(0), (224, 224))[0]
    torchvision.utils.save_image(y.float()[1], './debug_dino_self_dis_func.png')
    

    f = MaskContrast_Inter().cuda()
    
    # feature after maxpool.
    x, x_aug, mask = img_t[:,0:1,...].repeat(1,64,1,1), img_t[:,0:1,...].repeat(1,64,1,1), y
    # Don't be serious, just for validation.
    x = x + torch.randn_like(x)
    print(f(x, x_aug, mask))