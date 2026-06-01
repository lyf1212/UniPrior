import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.init as init
import torch.nn.functional as F


class ContrastiveMLPConv(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, bottle_channel, 3, padding=1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(bottle_channel, bottle_channel, 3, padding=1)
		self.fc = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal_(self.fc.weight)
		init.kaiming_normal_(self.conv1.weight)
		init.kaiming_normal_(self.conv2.weight)

	def forward(self, x):
		x = self.relu(self.conv2(self.relu(self.conv1(x))))
		x = F.adaptive_avg_pool2d(x, (1,1))
		x = x.view(x.size(0), x.size(1))
		return self.fc(x)


class ContrastiveMLPFC(nn.Module):
	def __init__(self, in_channel, out_channel, bottle_channel=64):
		super().__init__()
		self.fc1 = nn.Linear((in_channel**3)//16384, bottle_channel)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(bottle_channel, out_channel)
		init.kaiming_normal(self.fc1.weight)
		init.kaiming_normal(self.fc2.weight)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.fc2(self.relu(self.fc1(x)))


class ContrastiveHead(nn.Module):
	def __init__(self, feat_out_channels, out_channel):
		super().__init__()
		self.MLPs = []
		for in_channel in feat_out_channels:
			self.MLPs.append(ContrastiveMLPConv(in_channel, out_channel))
		self.MLPs = nn.ModuleList(self.MLPs)
	
	def forward(self, feats, bp=True):
		outputs = []
		for feat, MLP in zip(feats, self.MLPs):
			if bp:
				outputs.append(MLP(feat))
			else:
				outputs.append(MLP(feat).detach())
		return outputs