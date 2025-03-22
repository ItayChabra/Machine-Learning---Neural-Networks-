import torch
from torch import nn
import numpy as np
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Linear(torch.nn.Module):
  def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    else:
        self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    self.weight = nn.Parameter(torch.rand([self.out_features, self.in_features]))
    if self.bias is not None:
      self.bias = nn.Parameter(torch.rand([self.out_features]))

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    return torch.matmul(input, torch.transpose(self.weight,0,1)) + self.bias

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None
      )

class BTU(torch.nn.Module):
  def __init__(self, T=0.2, inplace: bool = False):
      super(BTU, self).__init__()
      self.T = T

  def forward(self, input: torch.Tensor) -> torch.Tensor:
      return 1 / (1 + torch.exp(-input/self.T))


dim = 2
out_dim = 1

class XOR_Net_Model(nn.Module):
    def __init__(self, num_hidden, bypass=True):
        super(XOR_Net_Model, self).__init__()
        self.bypass = bypass
        self.hidden = Linear(dim, num_hidden)  # Hidden layer with dynamic num_hidden
        if self.bypass:
            self.output = Linear(num_hidden + dim, out_dim)  # Concatenate input and hidden output
        else:
            self.output = Linear(num_hidden, out_dim)
        self.BTU = BTU(0.5)  # Example of BTU instance

    def forward(self, input):
        z1 = self.hidden(input)  # Hidden layer output
        y1 = self.BTU(z1)  # Apply BTU function
        if self.bypass:
            y1_concat = torch.cat((input, y1), 1)  # Concatenate input with hidden layer output
            z2 = self.output(y1_concat)  # Output layer
        else:
            z2 = self.output(y1)  # Output layer without bypass
        return self.BTU(z2)  # Apply BTU again to output


def Loss(out, t_train):
  return -torch.sum(t_train * torch.log(out) + (1.0 - t_train) * torch.log(1.0 - out))/out.size()[0]  # Cross Entropy loss function


stopping_threshold = 0.0001
patience = 10
max_epochs = 40000


def train(model, x_train, t_train, x_val, t_val, optimizer):
    validation_losses = []
    successful = False

    for epoch in range(max_epochs):
        # Training pass
        y_pred_train = model(x_train)
        train_loss = Loss(y_pred_train, t_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation pass
        y_pred_val = model(x_val)
        val_loss = Loss(y_pred_val, t_val).item()  # Compute validation loss

        # Track validation losses
        validation_losses.append(val_loss)

        # Stopping condition 1: check for improvement
        if len(validation_losses) > patience:
            recent_losses = validation_losses[-patience:]
            if max(recent_losses) - min(recent_losses) < stopping_threshold and val_loss < 0.2:
                successful = True
                break


    # Stopping condition 2: failed run
    failed = not successful and epoch == max_epochs - 1
    return {
        "successful": successful,
        "failed": failed,
        "train_loss": train_loss.item(),
        "val_loss": val_loss,
        "epochs": epoch + 1
    }

# Training set
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
t_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Validation set (training set + additional examples)
x_val = torch.tensor(
    [[0, 0], [0, 1], [1, 0], [1, 1], [1, 0.1], [1, 0.9], [0.9, 0.9], [0.1, 0.9]],
    dtype=torch.float32
)
t_val = torch.tensor(
    [[0], [1], [1], [0], [1], [0], [0], [1]],
    dtype=torch.float32
)

# Hyperparameter combinations
learning_rates = [0.1, 0.01]
hidden_neurons = [2, 4]
bypass_options = [True, False]
experiments = list(itertools.product(learning_rates, hidden_neurons, bypass_options))

experiments.append((0.01, 1, True))
results = []

def print_hidden_truth_table(model, x_train):
    """Prints the truth table for the hidden layer's output."""
    hidden_output = model.hidden(x_train)  # Get hidden layer output
    hidden_output = model.BTU(hidden_output).detach().numpy()  # Apply BTU and convert to numpy

    print("\nTruth Table (Hidden Layer Output):")
    print("Input A, Input B -> Hidden Neuron Output")
    for i in range(len(x_train)):
        print(f"{x_train[i].numpy()} -> {hidden_output[i]}")

# Run experiments 1–8 (without printing the truth table) and the 9th experiment
for i, (lr, hidden, bypass) in enumerate(experiments):
    print(f"\nRunning experiment {i + 1}: LR={lr}, Hidden={hidden}, Bypass={bypass}")
    successful_runs = 0
    failed_runs = 0
    metrics = {"epochs": [], "train_loss": [], "val_loss": []}

    final_model = None

    while successful_runs < 10:
        model = XOR_Net_Model(num_hidden=hidden, bypass=bypass)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Train the model
        run_result = train(model, x_train, t_train, x_val, t_val, optimizer)

        if run_result["successful"]:
            successful_runs += 1
            metrics["epochs"].append(run_result["epochs"])
            metrics["train_loss"].append(run_result["train_loss"])
            metrics["val_loss"].append(run_result["val_loss"])
            final_model = model  # Save the model after successful training

            # Print hidden truth table for the 9th experiment
            if hidden == 1:
                print(f"\nRun {successful_runs} (Hidden Layer Output for Experiment 9):")
                print_hidden_truth_table(final_model, x_train)
        elif run_result["failed"]:
            failed_runs += 1

    # Calculate averages and standard deviations
    avg_epochs = np.mean(metrics["epochs"])
    std_epochs = np.std(metrics["epochs"])
    avg_train_loss = np.mean(metrics["train_loss"])
    std_train_loss = np.std(metrics["train_loss"])
    avg_val_loss = np.mean(metrics["val_loss"])
    std_val_loss = np.std(metrics["val_loss"])

    # Save results
    results.append({
        "params": (lr, hidden, bypass),
        "avg_epochs": avg_epochs,
        "std_epochs": std_epochs,
        "avg_train_loss": avg_train_loss,
        "std_train_loss": std_train_loss,
        "avg_val_loss": avg_val_loss,
        "std_val_loss": std_val_loss,
        "failed_runs": failed_runs
    })


print("\nExperiment Results:")
for result in results:
    lr, hidden, bypass = result["params"]

    # Calculate percentage standard deviations
    std_epochs_percent = (result["std_epochs"] / result["avg_epochs"]) * 100 if result["avg_epochs"] != 0 else 0
    std_train_loss_percent = (result["std_train_loss"] / result["avg_train_loss"]) * 100 if result["avg_train_loss"] != 0 else 0
    std_val_loss_percent = (result["std_val_loss"] / result["avg_val_loss"]) * 100 if result["avg_val_loss"] != 0 else 0

    # Print results for each experiment
    print(f"LR={lr}, Hidden={hidden}, Bypass={bypass}")
    print(f"  Avg Epochs: {result['avg_epochs']:.2f} ± {result['std_epochs']:.2f} ({std_epochs_percent:.2f}%)")
    print(f"  Avg Train Loss: {result['avg_train_loss']:.4f} ± {result['std_train_loss']:.4f} ({std_train_loss_percent:.2f}%)")
    print(f"  Avg Val Loss: {result['avg_val_loss']:.4f} ± {result['std_val_loss']:.4f} ({std_val_loss_percent:.2f}%)")
    print(f"  Failed Runs: {result['failed_runs']}")
    print("\n")  # Added new line after each experiment