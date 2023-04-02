import torch 
from model import SimpleNN

def get_data(batch_size):
	x = torch.randn(batch_size, 1)
	y = 5 * x + 3

	return x, y

def train(model, optimizer, loss_function, epochs=100, batch_size=64):
	print("training model..")

	for i in range(epochs):

		optimizer.zero_grad()

		x, y = get_data(batch_size)
		y_pred = model(x)

		loss = loss_function(y_pred, y)

		loss.backward()
		optimizer.step()

		print(f"Epoch: {i}, Loss: {loss.item()}")
		# print(y_pred.shape)
		# print(y.shape)


if __name__ == '__main__':
	model = SimpleNN()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	loss_function = torch.nn.MSELoss()
	train(
		model=model, 
		optimizer=optimizer, 
		loss_function=loss_function,
		epochs=30000,
	)


	torch.save(model.state_dict(), '../models/simple_nn.pt')
	print("Model saved..")



