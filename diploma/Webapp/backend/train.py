import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import os

# Paths
dataset_path = "database/2data/cardio_train.csv"
scaler_path = "scaler.npy"
model_path = "best_model.pth"

# Load dataset
df = pd.read_csv(dataset_path, sep=';')

# Convert age from days to years
df['age'] = (df['age'] / 365).astype(int)

# Drop unnecessary columns
df.drop(columns=['id'], inplace=True)

# Split features and labels
X = df.drop(columns=['cardio']).values
y = df['cardio'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (Save or Load the Scaler)
if os.path.exists(scaler_path):
    print("✅ Loading existing scaler...")
    scaler_params = np.load(scaler_path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_params["mean"]
    scaler.scale_ = scaler_params["scale"]
    X_train = (X_train - scaler.mean_) / scaler.scale_
    X_test = (X_test - scaler.mean_) / scaler.scale_
else:
    print("❌ Scaler not found! Training a new model...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    np.save(scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})  # Save as .npy

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Compute sample weights for training
sample_weights = np.array([class_weights[int(label)] for label in y_train])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_data, batch_size=64, sampler=sampler)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Neural network model
class CardioNet(nn.Module):
    def __init__(self, input_dim):
        super(CardioNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # No Sigmoid here
        return x

# Initialize model, loss, optimizer, and scheduler
input_dim = X_train.shape[1]
model = CardioNet(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

best_acc = 0

# Check if a pre-trained model exists
if os.path.exists(model_path):
    print("✅ Loading pre-trained model...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    print("🚀 Training a new model...")
    early_stopping_counter = 0
    early_stopping_patience = 10
    
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluate on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        scheduler.step(acc)  # Adjust learning rate based on validation accuracy
        print(f"Epoch {epoch+1}/200, Loss: {epoch_loss/len(train_loader):.4f}, Test Acc: {acc*100:.2f}%")
        
        # Save the best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print("💾 New best model saved!")
            early_stopping_counter = 0  # Reset early stopping counter
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print("🛑 Early stopping activated.")
            break

# Load the best model for final evaluation
model.load_state_dict(torch.load(model_path))
model.eval()

# Final evaluation
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"✅ Final Test Accuracy: {correct / total * 100:.2f}%")
