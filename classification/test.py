import torch
from torch import nn
import argparse
import os
import torchvision
from torchvision import transforms
from utils import *
from codan import CODaN

parser = argparse.ArgumentParser(description='Zero Shot Day Night Domain Adaptation -- Classification')

parser.add_argument('--checkpoint', type=str, default='/mnt/netdisk/liyf/DAI-Net/DG/classification/checkpoints/0919_LOWAUG_0.3darken_wodinoadain_lrIIMdiv10_1e-4/model_best.pt',
					help='location for checkpoint')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')                    
parser.add_argument('--gpu_id', default='0', type=str)

# For model and training.
parser.add_argument('--dino_attn', type=float, default=0.1)
parser.add_argument('--reg_decode', type=float, default=0)
parser.add_argument('--reg_decode_tau', type=float, default=-1)
parser.add_argument('--iim_align', type=float, default=0.1)
parser.add_argument('--iim_align_tau', type=float, default=0.8, help='filter thres. in [0,1] in torch.quantile. -1 means no use.')
parser.add_argument('--mask_contrast', type=float, default=0.01)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def validate(model, test_night_loader):
	model.eval()  # without this, acc drop from 71.84 to 22.52!.
	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	top1 = AverageMeter()
	losses = AverageMeter()


	losses.reset()
	top1.reset()
	for images, labels in test_night_loader:
		images = images.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			outputs = model.test_forward(images)
			loss = criterion(outputs,labels)

		prec1 = accuracy(outputs.data, labels)[0]
		top1.update(prec1.item(), images.size(0))
		losses.update(float(loss.detach().cpu()))
	
	return top1.avg


def main():

	transforms_test = transforms.ToTensor()  # normalization is inside `forward`.
	test_night_dataset = CODaN(root='/mnt/netdisk/wangwenjing/Datasets/CODaN', split='full_night', transform=transforms_test)

	test_night_loader = torch.utils.data.DataLoader(test_night_dataset,
			num_workers=16,
			batch_size=args.batch_size,
			shuffle=False)


	state_dict = torch.load(args.checkpoint)

	from model import ResNet_IIMDG

	model = ResNet_IIMDG([2, 2, 2, 2], num_classes=10, args=args)

	# `strict=False`: some dino params are unnecessary and discard after training.
	model.load_state_dict(state_dict['state_dict'], strict=False)  
	model.cuda()

	acc = validate(model, test_night_loader)

	print('Nighttime accuracy: {:.2f}%'.format(acc))

if __name__ == '__main__':
	main()
