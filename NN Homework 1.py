import torch
from torch import nn
from torch import optim
import numpy as np
import itertools


class Linear(torch.nn.Module):
  def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    else:
        self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    self.weight = nn.Parameter(torch.rand([self.in_features, self.out_features]))
    if self.bias is not None:
      self.bias = nn.Parameter(torch.rand([self.out_features]))

  def set_weights(self, w, b):
      self.weight = nn.Parameter(w.clone().detach().requires_grad_(True))
      self.bias = nn.Parameter(b.clone().detach().requires_grad_(True))

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    return torch.matmul(input, self.weight) + self.bias # * is elementwise

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None
      )


class BTU(torch.nn.Module):
  def __init__(self, T):
      super(BTU, self).__init__()
      self.T = T

  def forward(self, input: torch.Tensor) -> torch.Tensor:
      return 1 / (1 + torch.exp(-input/self.T))


n = 2  # As given in the assignment
dim = 2  # 2^n neurons
out_dim = 1  # Single output
Temp = 0.001  # for sigmoid


class NeuronNet(nn.Module):
    def __init__(self, k: int, bypass: bool = True):
        super().__init__()
        self.bypass = bypass  # Flag to determine if bypass is used
        self.k = k  # Number of neurons in the hidden layer
        self.hidden = Linear(dim, k)  # Hidden layer with input dimension and k neurons
        if self.bypass:
            self.output = Linear(k + dim, out_dim)  # Concatenate input with hidden output if bypass is True
        else:
            self.output = Linear(k, out_dim)  # Simple output layer if bypass is False
        self.BTU = BTU(Temp)  # Temperature unit (BTU) for the network

    def forward(self, input):
        z1 = self.hidden(input)  # Pass input through hidden layer
        y1 = self.BTU(z1)  # Apply BTU function to hidden layer output

        if y1.dim() == 1:  # Ensure y1 has at least two dimensions
            y1 = y1.unsqueeze(1)  # Convert (batch_size,) to (batch_size, 1)

        if self.bypass:
            y1_concat = torch.cat((input, y1), 1)  # Concatenate input with y1 for bypass case
            z2 = self.output(y1_concat)  # Pass through output layer
        else:
            z2 = self.output(y1)  # Pass y1 through output layer if no bypass
        return self.BTU(z2)  # Return final output after applying BTU

    def set_weights(self, w, b, layerName):
        w_transposed = w.T  # Transpose weights to match expected shape
        if layerName == "hidden":
            expected_w_shape = self.hidden.weight.shape
            expected_b_shape = self.hidden.bias.shape
            if w_transposed.shape == expected_w_shape and b.shape == expected_b_shape:
                self.hidden.set_weights(w_transposed, b)  # Set weights for hidden layer if shapes match
            else:
                # Print error message if shapes don't match
                print(f"Error: Expected hidden layer weights shape {expected_w_shape}, "
                    f"but got {w_transposed.shape}. Expected hidden layer bias shape {expected_b_shape}, " f"but got {b.shape}.")
        elif layerName == "output":
            expected_w_shape = self.output.weight.shape
            expected_b_shape = self.output.bias.shape
            if w_transposed.shape == expected_w_shape and b.shape == expected_b_shape:
                self.output.set_weights(w_transposed, b)  # Set weights for output layer if shapes match
            else:
                # Print error message if shapes don't match
                print(f"Error: Expected output layer weights shape {expected_w_shape}, "
                    f"but got {w_transposed.shape}. Expected output layer bias shape {expected_b_shape}, " f"but got {b.shape}.")
        else:
            print("Invalid layer name. Enter 'hidden' or 'output'.")


def loss(model, x, y):
    y_pred = model(x)  # Forward pass to get predictions
    if y_pred.dim() == 1:  # Convert (batch_size,) to (batch_size, 1)
        y_pred = y_pred.unsqueeze(1)
    return torch.sum((y_pred - y) ** 2)  # Sum of squared errors


def main():
    # Input matrix
    x = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)

    # Expected output matrix for XOR
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    print("Input Matrix:\n", x)
    print("Expected Output:\n", y)
    print("\n" + "="*40 + "\n")

    # Part 1: k = 1, bypass = True
    hidWeight = nn.Parameter(torch.tensor([[1., 1.]]))
    hidBias = nn.Parameter(torch.tensor([-1.5]))
    outWeight = nn.Parameter(torch.tensor([[1., 1., -2.]]))
    outBias = nn.Parameter(torch.tensor([-0.5]))

    print("Testing: k = 1, bypass = True")
    net1 = NeuronNet(k=1, bypass=True) # Initialize the network
    net1.set_weights(hidWeight, hidBias, "hidden")
    net1.set_weights(outWeight, outBias, "output")

    # Print Hidden Layer Weights and Biases
    print("\nHidden Layer Weights (w):", hidWeight.data.numpy())
    print("Hidden Layer Biases (b):", hidBias.data.numpy())

    print("\nOutput Layer Weights (w):", outWeight.data.numpy())
    print("Output Layer Biases (b):", outBias.data.numpy())

    # Perform forward pass and compute loss
    output1 = net1.forward(x)
    print("\nOutput Matrix (truth table):")
    for i in range(len(x)):
        print(f"Input: {x[i]} => Output: {output1[i].item():.4f}")

    loss1 = loss(net1, x, y)
    print("Loss:", loss1.item())
    print("\n" + "="*40 + "\n")

    # Part 2: k = 2, bypass = False
    hidWeight = nn.Parameter(torch.tensor([[1., 1.], [-1., -1.]]))
    hidBias = nn.Parameter(torch.tensor([-0.5, 1.5]))
    outWeight = nn.Parameter(torch.tensor([[1., 1.]]))
    outBias = nn.Parameter(torch.tensor([-1.5]))

    print("Testing: k = 2, bypass = False")
    net2 = NeuronNet(k=2, bypass=False) # Initialize the network
    net2.set_weights(hidWeight, hidBias, "hidden")
    net2.set_weights(outWeight, outBias, "output")

    # Print Hidden Layer Weights and Biases
    print("\nHidden Layer Weights (w):", hidWeight.data.numpy())
    print("Hidden Layer Biases (b):", hidBias.data.numpy())

    print("\nOutput Layer Weights (w):", outWeight.data.numpy())
    print("Output Layer Biases (b):", outBias.data.numpy())

    # Perform forward pass and compute loss
    output2 = net2.forward(x)
    print("\nOutput Matrix (truth table):")
    for i in range(len(x)):
        print(f"Input: {x[i]} => Output: {output2[i].item():.4f}")

    loss2 = loss(net2, x, y)
    print("Loss:", loss2.item())
    print("\n" + "="*40 + "\n")

    # Part 3: k = 4, bypass = False
    hidWeight = nn.Parameter(torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]))
    hidBias = nn.Parameter(torch.tensor([0.5, -0.5, -0.5, -1.5]))
    outWeight = nn.Parameter(torch.tensor([[0., 1., 1., 0.]]))
    outBias = nn.Parameter(torch.tensor([-0.5]))

    print("Testing: k = 4, bypass = False")
    net3 = NeuronNet(k=4, bypass=False) # Initialize the network
    net3.set_weights(hidWeight, hidBias, "hidden")
    net3.set_weights(outWeight, outBias, "output")

    # Print Hidden Layer Weights and Biases
    print("\nHidden Layer Weights (w):", hidWeight.data.numpy())
    print("Hidden Layer Biases (b):", hidBias.data.numpy())

    print("\nOutput Layer Weights (w):", outWeight.data.numpy())
    print("Output Layer Biases (b):", outBias.data.numpy())

    # Perform forward pass and compute loss
    output3 = net3.forward(x)
    print("\nOutput Matrix (truth table):")
    for i in range(len(x)):
        print(f"Input: {x[i]} => Output: {output3[i].item():.4f}")

    loss3 = loss(net3, x, y)
    print("Loss:", loss3.item())
    print("\n" + "="*40 + "\n")


if __name__ == "__main__":
    main()
