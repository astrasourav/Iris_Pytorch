import torch
from model import Model

# Load trained model
model = Model()
model.load_state_dict(torch.load('Iris_Model.pt'))
model.eval()

# Example new data to predict
new_data = torch.tensor([5.9, 3.0, 5.1, 1.8])  # Example input

with torch.no_grad():
    output = model(new_data)
    predicted_class = torch.argmax(output).item()

print("Raw output:", output)
print("Predicted Class:", predicted_class)  # 0, 1, or 2
