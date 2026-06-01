python train.py --checkpoint /mnt/netdisk/liyf/DAI-Net/DG/classification/resnet18-5c106cde.pth \
    --epochs 20 \
    --experiment UniPrior_classification \
    --iim_align 2 \
    --dino_attn 0.1 \
    --mask_contrast 0.01 \
    --resume_from_resnet \
    --scheduler cosine \
    --darkness_ratio 0.3 \
