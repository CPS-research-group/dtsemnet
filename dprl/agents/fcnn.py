'''FCNN policy and value networks for PPO.'''
import torch
import torch.nn as nn
import torch.nn.init as init

class FCNN(nn.Module):
    """
    A fully connected neural network (FCNN) with optional policy/value outputs.

    Parameters:
    - input_dim (int): Dimension of the input features.
    - hidden_layers (list of int): List containing the number of units in each hidden layer.
    - output_dim (int): Dimension of the output layer.
    - policy (bool): If True, applies a softmax activation to the output (policy network).
                     If False, adds an extra linear layer for value prediction.
    """
    
    def __init__(self, input_dim, hidden_layers, output_dim, policy=True):
        super(FCNN, self).__init__()

        self.policy = policy
        self.layers = nn.ModuleList()

        # Input layer
        self._add_layer(input_dim, hidden_layers[0])

        # Hidden layers
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self._add_layer(in_dim, out_dim)

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        init.orthogonal_(self.layers[-1].weight)

        # Define output behavior
        if policy:
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.value_output = nn.Linear(output_dim, 1)
            init.orthogonal_(self.value_output.weight)

    def _add_layer(self, in_dim, out_dim):
        """Helper function to add a linear layer with orthogonal initialization and ReLU activation."""
        layer = nn.Linear(in_dim, out_dim)
        init.orthogonal_(layer.weight)
        self.layers.append(layer)
        self.layers.append(nn.ReLU())

    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)

        return self.output_activation(x) if self.policy else self.value_output(x)


if __name__ == "__main__":
    # Example usage:
    input_dim = 4
    output_dim = 2
    hidden_layers = [32, 32]

    policy_model = FCNN(input_dim, output_dim, hidden_layers, policy=True)
    value_model = FCNN(input_dim, output_dim, hidden_layers, policy=False)

    print("Policy model:")
    print(policy_model)
    print("\nValue model:")
    print(value_model)

    # Test the models with random input
    x = torch.randn(1, input_dim)
    policy_output = policy_model(x)
    value_output = value_model(x)
    print(f"\nPolicy output shape: {policy_output.shape}")
    print(f"Value output shape: {value_output.shape}")
