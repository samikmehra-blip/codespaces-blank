import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import SimpleCNN

class SPHERExDataset(Dataset):
    """Custom Dataset wrapper to ingest your .npy spatial cutouts and labels."""
    def __init__(self, data_array, labels_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.labels = torch.tensor(labels_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting compute device: {device}")

    print("Initializing dataset with tensor structures...")
    X_dummy = np.random.randn(200, 3, 64, 64).astype(np.float32) 
    y_dummy = np.random.randint(0, 2, size=(200, 1)).astype(np.float32)  

    dataset = SPHERExDataset(X_dummy, y_dummy)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()                    
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    epochs = 10
    print(f"Starting training run for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_x)
        
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    print("Pipeline execution successful! Ready to swap dummy data for your real SPHEREx arrays.")

if __name__ == "__main__":
    main()