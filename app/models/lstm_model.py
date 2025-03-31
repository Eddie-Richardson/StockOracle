# app/models/lstm_models.py
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out

def predict_stock_trend(data, forecast_steps=5):
    try:
        closing_prices = np.array([row["Close"] for row in data]).reshape(-1, 1)

        # Scale data between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_prices)

        # Prepare sequences for training
        sequence_length = 10
        if len(scaled_data) <= sequence_length:
            raise ValueError("Not enough data to create training sequences. Please provide more data.")
        X_train, y_train = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train).float().unsqueeze(2)
        y_train = torch.from_numpy(y_train).float().unsqueeze(1)

        # Initialize and train the LSTM model
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Prepare data for prediction
        model.eval()
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = last_sequence.reshape(1, sequence_length, 1)  # Batch size = 1, feature size = 1
        last_sequence = torch.from_numpy(last_sequence).float()

        # Predict future steps
        predictions = []
        for _ in range(forecast_steps):
            with torch.no_grad():
                pred = model(last_sequence)  # Output is [1, 1]
                pred = pred.view(1, 1, 1)  # Reshape to [batch_size, time_step, features]
                predictions.append(pred.item())
                last_sequence = torch.cat((last_sequence[:, 1:, :], pred), dim=1)

        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Evaluate metrics on in-sample predictions
        train_predictions = model(X_train).detach().numpy()
        train_predictions = scaler.inverse_transform(train_predictions)
        y_train_original = scaler.inverse_transform(y_train.numpy())
        mae = float(np.mean(np.abs(train_predictions - y_train_original)))
        mse = float(np.mean((train_predictions - y_train_original) ** 2))

        return {
            "predictions": predictions.tolist(),
            "metrics": {
                "MAE": mae,
                "MSE": mse
            }
        }
    except Exception as e:
        raise ValueError(f"Error fitting LSTM model: {e}")
