import torch 
from train.model import SimpleNN


if __name__ == "__main__":
	model = SimpleNN()
	torch_path = "./models/simple_nn.pt"
	onnx_path = "./models/simple_nn2.onnx"

	model.load_state_dict(torch.load(torch_path))
	input = torch.Tensor([[1],[2],[3]])
	output = 5 * input + 3

	pred = model(input) 
	print(pred)

	assert torch.max(torch.abs(pred-output)) < 1e-1

	dummy_input = torch.randn(1, 1)
	torch.onnx.export(model, dummy_input, onnx_path)
	print(f"Exported torch model at {torch_path} to ONNX format at: {onnx_path}")