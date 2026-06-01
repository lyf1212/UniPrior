import torch
import torch.nn as nn
import sys
sys.path.append('..')
from tools.illumination_invariants import IIM_kubelka
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


class OutlierRobustL1Loss(nn.Module):
    def __init__(self, tau=0.9):
        super().__init__()
        if not (0 < tau <= 1):
            raise ValueError(f"tau must in [0,1]!.")
        self.tau = tau

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_pixel_errors = torch.abs(prediction - target)

        if self.tau == 1.0:
            reliable_errors = per_pixel_errors
        else:
            threshold = torch.quantile(per_pixel_errors.flatten(1), self.tau, dim=1)[:, None, None, None]
            # print(threshold.shape)
            reliable_mask = per_pixel_errors <= threshold
            reliable_errors = per_pixel_errors[reliable_mask]

        loss = torch.mean(reliable_errors)

        return loss
    

class ResNet_IIMDG(nn.Module):

    def __init__(self, layers, args, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_IIMDG, self).__init__()

        '''
            IIM controller starts.
        '''

        self.IIM = IIM_kubelka()
        self.iim_channel = 6

        '''
            Basic model construction.
        '''
        block = BasicBlock
        self.args = args
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3+self.iim_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        '''
            Backbone.
        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        

        '''
            Initialization setting starts.
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)


        '''
            Head.
        '''
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        '''
            Decoder Head only for regularization.
        '''
        self.reg_decoder = None
        if args.reg_decode != 0:
            class LightDecoderBlock(nn.Module):
                def __init__(self, in_channels, skip_channels, out_channels):
                    super().__init__()
                    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    
                    self.conv = nn.Sequential(
                        nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                    nn.init.kaiming_normal_(self.conv[0].weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(self.conv[1].weight, 1)
                    nn.init.constant_(self.conv[1].bias, 0)

                def forward(self, x, skip_features):
                    x = self.upsample(x)
                    x = torch.cat([x, skip_features], dim=1)
                    x = self.conv(x)
                    return x

            class RegDecoder(nn.Module):
                def __init__(self, out_channels=6):
                    super().__init__()
                    
                    encoder_channels = {'layer4': 512, 'layer3': 256, 'layer2': 128, 'layer1': 64, 'conv1': 64}
                    
                    decoder_channels = {'block4': 128, 'block3': 64, 'block2': 32, 'block1': 16}

                    self.decoder_block4 = LightDecoderBlock(encoder_channels['layer4'], encoder_channels['layer3'], decoder_channels['block4'])
                    self.decoder_block3 = LightDecoderBlock(decoder_channels['block4'], encoder_channels['layer2'], decoder_channels['block3'])
                    self.decoder_block2 = LightDecoderBlock(decoder_channels['block3'], encoder_channels['layer1'], decoder_channels['block2'])
                    self.decoder_block1 = LightDecoderBlock(decoder_channels['block2'], encoder_channels['conv1'], decoder_channels['block1'])

                    self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    self.final_conv = nn.Conv2d(decoder_channels['block1'], out_channels, kernel_size=1)
                    
                def forward(self, features):
                    x = self.decoder_block4(features['layer4'], features['layer3'])
                    x = self.decoder_block3(x, features['layer2'])
                    x = self.decoder_block2(x, features['layer1'])
                    x = self.decoder_block1(x, features['conv1'])
                    
                    x = self.final_upsample(x)
                    x = self.final_conv(x)

                    return x

            self.reg_decoder = RegDecoder(self.iim_channel)
            if args.reg_decode_tau != -1:
                self.reg_decode_loss = OutlierRobustL1Loss(args.reg_decode_tau)
            else:
                self.reg_decode_loss = F.mse_loss

                
        '''
            Prior Consistency with outlier-robust loss.
        '''


        if self.args.iim_align != 0:
            if self.args.iim_align_tau != -1:
                self.iim_align_loss = OutlierRobustL1Loss(args.iim_align_tau)
            else:
                self.iim_align_loss = F.mse_loss

        if args.dino_attn != 0 or args.mask_contrast != 0:
            from tools.dinov2 import vit_base
            IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
            self.dino_preprocess = torchvision.transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            dino_model = vit_base(
                patch_size=14,
                img_size=518,  # 14*37.
                init_values=1.0,
                block_chunks=0)  # block_chunks: (int) split block sequence into block_chunks units for FSDP wrap.
            with torch.no_grad():
                from tools.path_register import DINO_PATH
                pretrained_weight = torch.load(DINO_PATH)
                dino_model.load_state_dict(pretrained_weight)
                dino_model.eval()
                self.dino_extract = dino_model.cuda()
        

        '''
            DINO Attention Map Alignment.
            Inspired by Zhiwei Hao, Jianyuan Guo, Ding Jia, Kai Han, Yehui Tang, Chao Zhang, Han Hu, and Yunhe Wang. Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation. NeurIPS'22.
        '''
        if args.dino_attn != 0:
            from tools.dino_loss import L_manifold
            self.dino_loss = L_manifold()

            
        '''
            DINO-based contrasitive learning.
        '''
        if args.mask_contrast != 0:
            from tools.img_contrasitive import MaskContrast_Inter, generate_cluster_mask
            self.mask_contra_loss = MaskContrast_Inter(patch_size=7)
            self.generate_cluster_mask = generate_cluster_mask

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    

    def test_forward(self, x):
        x_iim = self.IIM(x)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        x = TF.normalize(x, mean=mean, std=std)

        x = self.conv1(torch.concat([x, x_iim], dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ####################################################################################################

        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        f = self.fc(x)

        return f
    

    def forward(self, x, x_light):
        '''
            stem_layer      ->  | 64,56,56 | ->  | 128,28,28 | -> | 256,14,14 | -> | 512,7,7 | -> head.
          (IIM consistency)          mask_contrast    
        '''
        x_iim = self.IIM(x)

        loss_iim_align = 0
        if self.args.iim_align != 0:
            x_light_iim = self.IIM(x_light)
            loss_iim_align = self.args.iim_align * self.iim_align_loss(x_iim, x_light_iim.detach())

        if self.args.reg_decode != 0:
            x_light_iim = self.IIM(x_light).detach()  # gt of reg decoder.

        
        # print(torch.isnan(x).any(), torch.isnan(x_light).any())
        
        if self.args.dino_attn != 0:
            with torch.no_grad():
                # off-the-shelf DINOv2 for robust feature extraction.
                h_, w_ = x.shape[-2] // 14 * 14, x.shape[-1] // 14 * 14
                x_for_dino = self.dino_preprocess(F.interpolate(x, size=(h_, w_))) # 518=14*37.
                # torchvision.utils.save_image(x_for_dino, './debug_x_for_dino.png')
                dino_feat = self.dino_extract.get_intermediate_layers(x_for_dino)[0]

        if self.args.mask_contrast != 0:
            with torch.no_grad():
                # off-the-shelf DINOv2 for robust feature extraction.
                x_light_for_dino = self.dino_preprocess(F.interpolate(x_light, size=(518, 518))) # 518=14*37.
                mask_cluster = self.generate_cluster_mask(self.dino_extract, x_light_for_dino, 37)
                # torchvision.utils.save_image(mask_cluster, './debug_mask_cluster.png')

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        x = TF.normalize(x, mean=mean, std=std)


        x = self.conv1(torch.concat([x, x_iim], dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        # prepare for decode-regularization.
        if self.reg_decoder is not None:
            mid_features = {'conv1': x}
        x = self.maxpool(x)

        ####################################################################################################

        x = self.layer1(x)

        if self.reg_decoder is not None:
            mid_features['layer1'] = x

        loss_dino = 0
        if self.args.dino_attn != 0:
            loss_dino = self.args.dino_attn * self.dino_loss(x, dino_feat.detach())

        if self.args.mask_contrast != 0:
            x_light_iim = self.IIM(x_light)
            x_light = TF.normalize(x_light, mean=mean, std=std)
            x_light = self.conv1(torch.concat([x_light, x_light_iim], dim=1))
            x_light = self.bn1(x_light)
            x_light = self.relu(x_light)
            x_light = self.maxpool(x_light)
            x_light = self.layer1(x_light)

        loss_mask_contrast = 0
        if self.args.mask_contrast != 0:
            # x_vis = (x - x.min()) / (x.max() - x.min())
            # print(x_vis.shape, x_vis.dtype)
            # print(x_light.mean(), x_light.std(), x.mean(), x.std())
            # x_light_vis = (x_light - x_light.min()) / (x_light.max() - x_light.min())
            # torchvision.utils.save_image(x_vis.mean(1, keepdim=True), 'debug_x.png')
            # torchvision.utils.save_image(x_light_vis.mean(1, keepdim=True), 'debug_x_light.png')
            loss_mask_contrast = self.args.mask_contrast * self.mask_contra_loss(x, x_light.detach(), mask_cluster)
        
    
        ####################################################################################################

        x = self.layer2(x)

        if self.reg_decoder is not None:
            mid_features['layer2'] = x

        ####################################################################################################

        x = self.layer3(x)

        if self.reg_decoder is not None:
            mid_features['layer3'] = x

        ####################################################################################################
        
        x = self.layer4(x)

        if self.reg_decoder is not None:
            mid_features['layer4'] = x

        ####################################################################################################
                    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        f = self.fc(x)

        ############################## Regularization Decoding #############################################

        loss_decode = 0
        if self.reg_decoder is not None:
            decode_iim = self.reg_decoder(mid_features)
            # print(self.reg_decode_loss, x_light_iim.shape)
            loss_decode = self.args.reg_decode * self.reg_decode_loss(x_light_iim, decode_iim)

        return f, loss_iim_align, loss_mask_contrast, loss_decode, loss_dino



