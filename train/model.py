import torch 
import torch.nn as nn


class SimpleNN(nn.Module):

	def __init__(self):
		super().__init__()
		self.layer = nn.Linear(1,1)

	def forward(self, x):
		return self.layer(x)



if __name__ == '__main__':
	model = SimpleNN()

	x = torch.randn(10, 1)
	y_pred = model(x)

	print(y_pred.shape)

