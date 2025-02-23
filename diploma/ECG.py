import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
import scipy.io
import os

# ---------------------- 1. Load and Preprocess ECG Data ---------------------- #
def butter_lowpass_filter(data, cutoff=50, fs=500, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

class ECGDataset(Dataset):
    def __init__(self, data_folder, label_csv, seq_length=5000):
        self.data_folder = data_folder
        self.labels = pd.read_csv(label_csv)
        self.files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
        self.label_dict = {"Normal": 0, "AFIB": 1, "PVC": 2, "RBBB": 3, "LBBB": 4}
        self.seq_length = seq_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.files[idx])
        mat_data = scipy.io.loadmat(file_path)
        ecg_signal = mat_data['val'][0][:self.seq_length] 
        ecg_signal = butter_lowpass_filter(ecg_signal)
        label = self.label_dict[self.labels.iloc[idx, 1]]
        return torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# ---------------------- 2. Load and Preprocess Medical Data ---------------------- #
class MedicalDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.labels = torch.tensor(self.data['cardio'].values, dtype=torch.long)
        self.features = torch.tensor(self.data.drop(columns=['cardio']).values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ---------------------- 3. Define Models ---------------------- #
class ECG_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=5):
        super(ECG_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class MedicalMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super(MedicalMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------- 4. Training Function ---------------------- #
def train_model(model, dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

# ---------------------- 5. Train and Save Models ---------------------- #
# Train ECG Model
ecg_dataset = ECGDataset("2data", "database/1data/cardio_train.csv")
ecg_dataloader = DataLoader(ecg_dataset, batch_size=32, shuffle=True)
ecg_model = ECG_LSTM()
ecg_criterion = nn.CrossEntropyLoss()
ecg_optimizer = optim.Adam(ecg_model.parameters(), lr=0.001)
train_model(ecg_model, ecg_dataloader, ecg_criterion, ecg_optimizer)
torch.save(ecg_model.state_dict(), "ecg_model.pth")

# Train Medical Model
medical_dataset = MedicalDataset("database/1data/cardio_train.csv")
medical_dataloader = DataLoader(medical_dataset, batch_size=32, shuffle=True)
medical_model = MedicalMLP(input_size=11)
medical_criterion = nn.CrossEntropyLoss()
medical_optimizer = optim.Adam(medical_model.parameters(), lr=0.001)
train_model(medical_model, medical_dataloader, medical_criterion, medical_optimizer)
torch.save(medical_model.state_dict(), "medical_model.pth")

# ---------------------- 6. Identify Cardiovascular Disease ---------------------- #
def identify_patient(ecg_signal, medical_features, ecg_model, medical_model, device='cuda'):
    ecg_model.to(device).eval()
    medical_model.to(device).eval()
    
    ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    medical_features = torch.tensor(medical_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ecg_pred = torch.softmax(ecg_model(ecg_signal), dim=1).cpu().numpy()
        medical_pred = torch.softmax(medical_model(medical_features), dim=1).cpu().numpy()
    
    combined_pred = (ecg_pred + medical_pred) / 2
    final_class = np.argmax(combined_pred)
    return final_class

# Example Usage
ecg_test_signal = np.random.randn(5000)  # Replace with real ECG data
medical_test_features = np.random.rand(11)  # Replace with real medical data

ec_model = ECG_LSTM()
ec_model.load_state_dict(torch.load("ecg_model.pth"))

med_model = MedicalMLP(input_size=11)
med_model.load_state_dict(torch.load("medical_model.pth"))

prediction = identify_patient(ecg_test_signal, medical_test_features, ec_model, med_model)
print("Predicted Class:", prediction)
