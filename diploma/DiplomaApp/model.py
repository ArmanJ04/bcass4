import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  

class CardioNet(nn.Module):
    def __init__(self, input_dim):
        super(CardioNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc5(x))
        return x

# Load the model
input_dim = 11  # Number of input features
model = CardioNet(input_dim)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

try:
    scaler = joblib.load("scaler.pkl")  # Load the scaler from a saved file
except FileNotFoundError:
    raise FileNotFoundError("Scaler file 'scaler.pkl' not found. Please ensure the scaler is saved during training.")

def identify_patient(data):
    try:
        # Ensure the input data is a numpy array of shape (1, 11)
        data = np.array(data).reshape(1, -1)
        if data.shape[1] != input_dim:
            raise ValueError(f"Input data must have {input_dim} features. Got {data.shape[1]} instead.")

        # Scale the input data
        scaled_data = scaler.transform(data)  # Use the pre-fitted scaler
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)

        # Make a prediction
        with torch.no_grad():
            prediction = model(tensor_data).item()

        # Return the predicted class (0 or 1)
        return int(prediction > 0.5)
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")