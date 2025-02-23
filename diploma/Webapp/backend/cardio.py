import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

dataset_path = "database/2data/cardio_train.csv"
scaler_path = "scaler.ny.npy"
model_path = "best_model.pth"

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
        x = self.fc4(x)
        return x

def train():
    print("Training started...")
    df = pd.read_csv(dataset_path, sep=';')
    df['age'] = (df['age'] / 365).astype(int)
    df.drop(columns=['id'], inplace=True)
    
    X = df.drop(columns=['cardio']).values
    y = df['cardio'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    if os.path.exists(scaler_path):
        scaler_params = np.load(scaler_path, allow_pickle=True).item()
        scaler.mean_, scaler.scale_ = scaler_params["mean"], scaler_params["scale"]
        X_train = (X_train - scaler.mean_) / scaler.scale_
        X_test = (X_test - scaler.mean_) / scaler.scale_
    else:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        np.save(scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    sample_weights = np.array([class_weights[int(label)] for label in y_train])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    input_dim = X_train.shape[1]
    model = CardioNet(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    best_acc = 0
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
        scheduler.step(acc)

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print("Training complete. Best Accuracy:", best_acc)

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CardioNet(input_dim=11).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(json.dumps({"error": "Model file not found. Train the model first."}))
        sys.exit(1)

    try:
        scaler_params = np.load(scaler_path, allow_pickle=True).item()
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = scaler_params["mean"], scaler_params["scale"]
    except FileNotFoundError:
        print(json.dumps({"error": "Scaler file not found. Train the model first."}))
        sys.exit(1)

    try:
        input_json = json.loads(sys.stdin.read())  
        if "features" not in input_json or not isinstance(input_json["features"], list):
            raise ValueError("Invalid input format. 'features' must be a list.")

        input_data = np.array(input_json["features"]).reshape(1, -1)
        input_data = scaler.transform(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor).item()

        probability = torch.sigmoid(torch.tensor(output)).item()
        
        print(json.dumps({"prediction": probability}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        train()
    else:
        predict()
