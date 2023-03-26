import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import torch

class Siamese_CelebA_Latent(nn.Module):
	def __init__(self, latent_dim=128, middle_features=1024, lr=1e-3, gamma=0.95):
		super(Siamese_CelebA_Latent, self).__init__()
		linear_layers = [
			nn.Linear(latent_dim, middle_features),
			nn.ReLU(),
			nn.Linear(middle_features, middle_features),
			nn.ReLU(),
			nn.Linear(middle_features, latent_dim),
		]
		self.vector_maker = nn.Sequential(*linear_layers)
		self.loss = nn.TripletMarginLoss()
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	def forward(self, anchor, pos, neg):
		anchor = self.vector_maker(anchor)
		pos = self.vector_maker(pos)
		neg = self.vector_maker(neg)
		return anchor, pos, neg
	
	def loss_function(self, anchor, pos, neg):
		return self.loss(anchor, pos, neg)

	def find_accuracy(self, anchor, pos, neg):
		d = lambda a,b: torch.sum((a-b)**2, axis=-1)
		return (d(anchor,pos) > (d(anchor,neg) + 1)).sum(axis=0).item() / len(anchor)

	def save(self, path):
		torch.save({
			'model':self.state_dict(),
			'optimizer':self.optimizer.state_dict(),
			'scheduler':self.scheduler.state_dict()
		}, path)

	def load(self, path):
		checkpoint = torch.load(path)
		self.load_state_dict(checkpoint['model'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.scheduler.load_state_dict(checkpoint['scheduler'])

	def training_step(self, batch):
		anchor, pos, neg = batch
		anchor, pos, neg = anchor.to(self.device), pos.to(self.device), neg.to(self.device)
		self.optimizer.zero_grad()
		anchor_f, pos_f, neg_f = self.forward(anchor, pos, neg)
		loss = self.loss_function(anchor_f, pos_f, neg_f)
		acc = self.find_accuracy(anchor_f, pos_f, neg_f)
		loss.backward()
		self.optimizer.step()
		return loss.item(), acc

	def val_step(self, batch):
		with torch.no_grad():
			anchor, pos, neg = batch
			anchor, pos, neg = anchor.to(self.device), pos.to(self.device), neg.to(self.device)
			anchor_f, pos_f, neg_f = self.forward(anchor, pos, neg)
			loss = self.loss_function(anchor_f, pos_f, neg_f)
			acc = self.find_accuracy(anchor_f, pos_f, neg_f)
		return loss.item(), acc