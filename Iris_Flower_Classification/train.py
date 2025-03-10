import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

def train_model():
    # Set random seed
    torch.manual_seed(32)

    # Load data
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    df = pd.read_csv(url)
    df['variety'] = df['variety'].replace({'Setosa': 0.0, 'Versicolor': 1.0, 'Virginica': 2.0}).astype(float)

    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)

    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Initialize model
    model = Model().to(device)

    # Loss and optimizer
    loss_fun = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    losses = []

    for epoch in range(epochs):
        y_pred = model(x_train)
        loss = loss_fun(y_pred, y_train)
        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Save model
    torch.save(model.state_dict(), 'Iris_Model.pt')
    print("Model saved as Iris_Model.pt")

    # Plot loss curve
    plt.plot(range(epochs), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('loss.png')
    print("Loss curve saved as loss.png")

    # Test accuracy
    with torch.no_grad():
        y_eval = model(x_test)
        correct = (torch.argmax(y_eval, dim=1) == y_test).sum().item()
        total = len(y_test)
        acc = (correct / total) * 100
        print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    train_model()
