import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardizing the dataset

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Save train and test data to CSV files
train_data = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
train_data['target'] = y_train  # Add target column
train_data.to_csv('bank-note_2/train.csv', index=False)

test_data = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
test_data['target'] = y_test  # Add target column
test_data.to_csv('bank-note_2/test.csv', index=False)


# Define the neural network class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_function, output_size=2):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # Add hidden layers
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_layers))
            if activation_function == 'tanh':
                layers.append(nn.Tanh())
            elif activation_function == 'relu':
                layers.append(nn.ReLU())
            prev_size = hidden_layers

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

        # Initialize weights
        if activation_function == 'tanh':
            # Xavier initialization
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
        elif activation_function == 'relu':
            # He initialization
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        return self.model(x)


# Training function
def train_model(model, X_train, y_train, X_test, y_test, epochs=2, batch_size=32):
    # Use Adam optimizer and CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on training and test set
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).argmax(dim=1)
            test_preds = model(X_test).argmax(dim=1)
            train_error = accuracy_score(y_train, train_preds)
            test_error = accuracy_score(y_test, test_preds)

        print(f'Epoch {epoch + 1}/{epochs} | Train error: {train_error:.6f} | Test error: {test_error:.6f}')

    return model


# Experiment parameters
depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = ['tanh', 'relu']

# Run experiments for each combination of depth, width, and activation function
for activation in activations:
    print(f"\nActivation Function: {activation}")
    for depth in depths:
        for width in widths:
            print(f"depth={depth}, width={width}")
            model = MLP(input_size=20, hidden_layers=width, activation_function=activation)
            model = train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
