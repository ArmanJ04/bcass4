import pandas as pd
import torch

class MedicalDataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.labels = torch.tensor(self.data['cardio'].values, dtype=torch.long)
        self.features = torch.tensor(self.data.drop(columns=['cardio']).values, dtype=torch.float32)

    def get_patient_data(self, idx):
        return self.features[idx], self.labels[idx]
