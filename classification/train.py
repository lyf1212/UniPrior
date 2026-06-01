import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn
import argparse
import os
import time
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from utils import *
from codan import CODaN
from tqdm import tqdm
from torch.backends import cudnn
import random
import torchvision.transforms.functional as TF
import gc

parser = argparse.ArgumentParser(description='Zero Shot Day Night Domain Adaptation -- Classification')
parser.add_argument('--experiment', type=str,
					help='location for saving trained models')
# parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume_from_resnet', action='store_true')
parser.add_argument('--checkpoint', type=str, default='/mnt/netdisk/liyf/DAI-Net/DG/classification/resnet18-5c106cde.pth',
					help='location for pre-trained daytime model')
parser.add_argument('--epochs', default=100, type=int,
					help='number of total epochs to run')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')                    
parser.add_argument('--lr', default=1e-4, type=float, help='optimizer lr')
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--decreasing_lr',default='50,120', type=str, help='decreasing lr at which epoch')
parser.add_argument('--scheduler', default='step', type=str, help='multistep or linear or cosine')
parser.add_argument('--gpu_id', default='0', type=str)

# For model and training.
parser.add_argument('--dino_attn', type=float, default=0.1)
parser.add_argument('--reg_decode', type=float, default=0)
parser.add_argument('--reg_decode_tau', type=float, default=-1)
parser.add_argument('--iim_align', type=float, default=0.1)
parser.add_argument('--iim_align_tau', type=float, default=0.8, help='filter thres. in [0,1] in torch.quantile. -1 means no use.')
parser.add_argument('--mask_contrast', type=float, default=0.01)
parser.add_argument('--use_gamma_noise', action='store_true', default=False, help='use naive illumination perturbation but not darkisp')
parser.add_argument('--noisy_iters', type=int, default=1)
parser.add_argument('--darkness_ratio', type=float, default=0.2)


args = parser.parse_args()
writer = SummaryWriter(os.path.join('checkpoints', args.experiment,'tensorboard'))

seed = 1008
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import numpy as np
np.random.seed(seed)
random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
save_dir = os.path.join('checkpoints', args.experiment)
if os.path.exists(save_dir) is not True:
	os.system("mkdir -p {}".format(save_dir))
log = logger(path=save_dir)
log.info(str(args))


def train(model, train_loader, optimizer, scheduler, epoch, log, darkening_model=None):
	model.train()
	loss_ce_day_rec, loss_ce_night_rec, loss_mutual_rec, loss_iim_rec, loss_mask_contrast_rec, loss_decode_rec, loss_dino_rec, losses_total =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
	criterion = torch.nn.CrossEntropyLoss()
	st = time.time()


	for i, (images, labels) in enumerate(tqdm(train_loader)):
		images = images.cuda()
		labels = labels.cuda()
		mask_zero = (images == 0)
		images = torch.clamp(images, 1e-5, 1)


		with torch.no_grad():
			night_images = darkening_model(images)
			# print(night_images, images)
			
			night_images[mask_zero] = 0
			night_images = torch.clamp(night_images, 1e-5, 1)



		outputs_day = model.test_forward(images)
		loss_ce_day = criterion(outputs_day, labels)

		outputs, loss_iim, loss_mask_contrast, loss_decode, loss_dino = model(night_images, images)
		
		loss_ce_night = criterion(outputs, labels)
		# print(outputs, labels)
		loss = loss_ce_day + loss_ce_night + loss_iim + loss_mask_contrast + loss_decode
		# print(torch.norm(torch.autograd.grad(loss_ce_night, model.conv1.weight, retain_graph=True)[0]))  # 6.10.
		# print(torch.norm(torch.autograd.grad(loss_mutual, model.conv1.weight, retain_graph=True)[0]))  # 0.11.
		# print(torch.norm(torch.autograd.grad(loss_mask_contrast, model.conv1.weight, retain_graph=True)[0]))  # 0.03.


		# Update recordings.
		loss_ce_day_rec.update(float(loss_ce_day.detach().cpu()), labels.shape[0])
		loss_ce_night_rec.update(float(loss_ce_night.detach().cpu()), labels.shape[0])
		loss_iim_rec.update(float(loss_iim.detach().cpu()), labels.shape[0])
		if args.mask_contrast != 0:
			loss_mask_contrast_rec.update(float(loss_mask_contrast.detach().cpu()), labels.shape[0])
		if args.reg_decode != 0:
			loss_decode_rec.update(float(loss_decode.detach().cpu()), labels.shape[0])
		if args.dino_attn != 0:
			loss_dino_rec.update(float(loss_dino.detach().cpu()), labels.shape[0])
		losses_total.update(float(loss.detach().cpu()), labels.shape[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_time = time.time() - st
	log.info(f'Epoch: {epoch}\t  Time: {train_time:.2f}')
	log.info(f'Loss_All: {losses_total.avg:.4f}\t \
			   Loss_CE_day: {loss_ce_day_rec.avg:.4f} \t \
			   Loss_CE: {loss_ce_night_rec.avg:.4f} \t ')
	writer.add_scalar('loss/total', losses_total.avg, epoch)
	writer.add_scalar('loss/ce_day', loss_ce_day_rec.avg, epoch)
	writer.add_scalar('loss/ce', loss_ce_night_rec.avg, epoch)
	writer.add_scalar('loss/loss_iim', loss_iim_rec.avg, epoch)
	if args.mask_contrast != 0:
		writer.add_scalar('loss/loss_mask_contrast', loss_mask_contrast_rec.avg, epoch)
	if args.reg_decode != 0:
		writer.add_scalar('loss/loss_decode', loss_decode_rec.avg, epoch)
	if args.dino_attn != 0:
		writer.add_scalar('loss/loss_dino', loss_dino_rec.avg, epoch)
	scheduler.step()

def validate(model, log, test_night_loader, test_day_loader, darkening_model):
	model.eval()
	torch.cuda.empty_cache() 
	gc.collect()

	criterion = nn.CrossEntropyLoss()
	criterion = criterion.cuda()

	top1 = AverageMeter()
	losses = AverageMeter()

	acc = []
	
	for i, dataset in enumerate([test_day_loader, test_night_loader]):
		losses.reset()
		top1.reset()
		for images, labels in dataset:
			images = images.cuda()
			labels = labels.cuda()		

			# torchvision.utils.save_image(images, f'./debug_val_input_enh.png')

			with torch.no_grad():
				outputs = model.test_forward(images)
				loss = criterion(outputs, labels)

			prec1 = accuracy(outputs.data, labels)[0]
			top1.update(prec1.item(), images.size(0))
			losses.update(float(loss.detach().cpu()))

		acc.append(top1.avg)

		if i == 0:
			log.info(f"Accuracy Day: {top1.avg:.2f}\t Loss: {losses.avg:.4f}")
		else:
			log.info(f"Accuracy Night: {top1.avg:.2f}\t Loss: {losses.avg:.4f}")

	return acc


def main():
	transforms_train = transforms.Compose([transforms.RandomResizedCrop(224,(0.8,1.0)),
								transforms.RandomHorizontalFlip(p=0.5),
								transforms.ToTensor(),
								])
	transforms_test = transforms.Compose([transforms.ToTensor(),
								# transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),  # abolished.
        ])

	import sys
	sys.path.append('..')
	from tools.path_register import CODAN_ROOT

	train_dataset = CODaN(root=CODAN_ROOT, split='train', transform=transforms_train)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=4)
	
	test_day_dataset = CODaN(root=CODAN_ROOT, split='test_day', transform=transforms_test)
	test_night_dataset = CODaN(root=CODAN_ROOT, split='full_night', transform=transforms_test)

	test_night_loader = torch.utils.data.DataLoader(test_night_dataset,
			num_workers=0,
			batch_size=args.batch_size,
			shuffle=False)
	test_day_loader = torch.utils.data.DataLoader(test_day_dataset,
			num_workers=0,
			batch_size=args.batch_size,
			shuffle=False)

	start_epoch = 0
	assert args.checkpoint is not None, 'You have to appoint an initialization, ResNet18/trained_on_CODaN-day/resumed ckpt.'
	state_dict = torch.load(args.checkpoint, map_location='cpu')

	from model import ResNet_IIMDG
	model = ResNet_IIMDG([2, 2, 2, 2], num_classes=10, args=args)
	
	if args.resume_from_resnet:
		if 'state_dict' in state_dict:
			state_dict = state_dict['state_dict']
		state_dict_useful = {}
		# print(state_dict.keys())
		
		for k, v in state_dict.items():
			if not 'fc' in k:  # the final output dimension is different from original setting.
				state_dict_useful[k] = v
			
			if 'conv1' in k and '.conv1' not in k:  # initialize the first stem layer.
				dim_out, dim_in, ksize, ksize = v.shape
				new_weight = torch.concat([v, torch.zeros(dim_out, model.iim_channel, ksize, ksize)], dim=1)
				state_dict_useful[k] = new_weight

		model.load_state_dict(state_dict_useful, strict=False)

		# check for uninitilized params.
		for k, v in model.state_dict().items():
			if k not in state_dict_useful.keys():
				print('{} is not initialized from pretrained ckpt'.format(k))
	else:
		# resume mid ckpt.
		model.load_state_dict(state_dict['state_dict'], strict=True)
		start_epoch = state_dict['epoch']
	
	# for k, v in model.state_dict().items():
		# print(k, v.shape)

	model.cuda()

	# Darken Model.
	if args.use_gamma_noise:
		
		def naive_perturb(x):
			with torch.no_grad():
				x = x.clamp_(0,1)
				x = x ** 2.2
				noise = torch.randn_like(x) * 0.05
				x = (x + noise).clamp(1e-5, 1)
				m = torch.distributions.laplace.Laplace(0, 0.05)
				noise = m.sample(x.shape).to(x.device)
				x = (x + noise).clamp(1e-5, 1)
			return x
		
		darkisp = naive_perturb
	else:
		from tools.darkisp_batched_trainable import DarkISP
		darkisp = DarkISP(device='cuda').cuda()
		darkisp.requires_grad_(False)
		darkisp.noisy_iters = args.noisy_iters
		darkisp.darkness_ratio.data = torch.tensor([args.darkness_ratio], device='cuda')



	cudnn.benchmark = True
	param_group = []
	for name, p in model.named_parameters():
		if 'iim' in name:
			param_group.append({'params': p, 'lr': args.lr})
		elif 'IIM' in name:
			param_group.append({'params': p, 'lr': args.lr/10.})
		elif 'dino' not in name:
			param_group.append({'params': p, 'lr': args.lr})

	if args.optimizer == 'adam':
		optimizer = torch.optim.Adam(param_group, lr=args.lr,
										weight_decay=1e-5)
	elif args.optimizer == 'sgd':
		optimizer = torch.optim.SGD(param_group, lr=args.lr,
								momentum=0.9, weight_decay=1e-5)
	else:
		assert False
	
	if args.scheduler == 'linear':
		lambdalr = lambda epoch: 1 if (epoch < args.epochs // 2) else (1 - epoch / args.epochs)
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdalr)
	elif args.scheduler == 'step':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
				optimizer, milestones=[int(i) for i in args.decreasing_lr.split(',')], gamma=0.1)
	elif args.scheduler == 'cosine':
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
	else:
		assert False
	
	for _ in range(start_epoch):
		scheduler.step()

	best_acc = 0.0
 
	# For debug.
	acc = validate(model, log, test_night_loader, test_day_loader, darkisp)

	for epoch in range(start_epoch+1, args.epochs+1):
		log.info("current lr is {}".format(
			optimizer.state_dict()['param_groups'][0]['lr']))

		train(model, train_loader, optimizer, scheduler, epoch, log, darkening_model=darkisp)
		
	
		acc = validate(model, log ,test_night_loader, test_day_loader, darkisp)

		writer.add_scalar("Accuracy/test_day", acc[0], epoch)
		writer.add_scalar("Accuracy/test_night", acc[1], epoch)

		
		if best_acc < acc[0]:
			best_acc = acc[0]
			state_dict = model.state_dict()
			save_state_dict = {}
			for k, v in state_dict.items():
				if 'dino_extract' not in k:
					save_state_dict[k] = v
			save_checkpoint({
				'epoch': epoch,
				'state_dict': save_state_dict,
				'optim': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'best_acc': acc[1],
			}, filename=os.path.join(save_dir, 'model_best.pt'))
		
		if epoch % 5 == 0:
			state_dict = model.state_dict()
			save_state_dict = {}
			for k, v in state_dict.items():
				if 'dino_extract' not in k:
					save_state_dict[k] = v
			save_checkpoint({
				'epoch': epoch,
				'state_dict': save_state_dict,
				'optim': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
			}, filename=os.path.join(save_dir, f'model_{epoch}.pt'))
	
	log.info(f"Best accuracy: {best_acc:.2f}")

if __name__ == '__main__':
	main()
