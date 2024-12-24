import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


class TimeSeriesDataset(Dataset):
    """ A PyTorch dataset for multivariate time series forecasting. """

    def __init__(self, data, input_length, forecast_horizon):
        """ Initialize the dataset.
        Args:
            data (torch.Tensor): Multivariate time series data of shape (N, T, F),
                                 where N = number of samples, T = time steps, F = features.
            input_length (int): Length of the input sequence.
            forecast_horizon (int): Length of the forecasting horizon.
        """
        self.data = data
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.input_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx: idx + self.input_length]
        target_seq = self.data[idx + self.input_length: idx + self.input_length + self.forecast_horizon]
        return input_seq, target_seq


class TransformerForecaster(nn.Module):
    """
    Transformer-based model for probabilistic, multivariate forecasting.
    """

    def __init__(self, feature_dim, num_layers, num_heads, hidden_dim, dropout=0.1):
        """
        Initialize the forecaster.

        Args:
            feature_dim (int): Dimensionality of each input feature.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in each layer.
            hidden_dim (int): Hidden dimension of the feedforward layers.
            dropout (float): Dropout rate.
        """
        super(TransformerForecaster, self).__init__()
        self.feature_dim = feature_dim

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.output_layer = nn.Linear(feature_dim, feature_dim)

    def forward(self, input_seq):
        """
        Forward pass for the transformer.

        Args:
            input_seq (torch.Tensor): Input sequence of shape (T, N, F),
                                      where T = time steps, N = batch size, F = features.
        Returns:
            torch.Tensor: Forecasted sequence of shape (T, N, F).
        """
        encoded_seq = self.encoder(input_seq)
        return self.output_layer(encoded_seq)


class ProbabilisticForecaster:
    """
    A probabilistic forecaster wrapping the TransformerForecaster model.
    """

    def __init__(self, feature_dim, num_layers=4, num_heads=8, hidden_dim=256, dropout=0.1, learning_rate=1e-4):
        """
        Initialize the probabilistic forecaster.

        Args:
            feature_dim (int): Number of features in the input data.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension of the feedforward network.
            dropout (float): Dropout rate.
            learning_rate (float): Learning rate for training.
        """
        self.model = TransformerForecaster(feature_dim, num_layers, num_heads, hidden_dim, dropout)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, num_epochs=50, device="cpu"):
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of training epochs.
            device (str): Device to run the training on ('cpu' or 'cuda').
        """
        self.model.to(device)
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for input_seq, target_seq in train_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                self.optimizer.zero_grad()
                predictions = self.model(input_seq)
                loss = self.loss_fn(predictions[-1], target_seq[-1])  # Forecast the last step
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

    def predict(self, input_seq, device="cpu"):
        """
        Make a probabilistic prediction.

        Args:
            input_seq (torch.Tensor): Input sequence of shape (T, N, F).
            device (str): Device to run the prediction on ('cpu' or 'cuda').
        Returns:
            torch.Tensor: Predicted sequence of shape (T, N, F).
        """
        self.model.eval()
        self.model.to(device)
        input_seq = input_seq.to(device)
        with torch.no_grad():
            return self.model(input_seq)


# Example Usage
if __name__ == "__main__":

    # Generate synthetic data (N=1000, T=50, F=3)
    data = torch.randn(1000, 50, 3)

    # Create dataset and dataloader
    input_length = 30
    forecast_horizon = 10
    dataset = TimeSeriesDataset(data, input_length, forecast_horizon)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize forecaster
    forecaster = ProbabilisticForecaster(feature_dim=3)

    # Train the model
    forecaster.train(dataloader, num_epochs=10)

    # Make a prediction
    test_input = data[:input_length].unsqueeze(1)  # Add batch dimension
    prediction = forecaster.predict(test_input)
    print(prediction.shape)  # Should match forecast_horizon x batch_size x feature_dim
