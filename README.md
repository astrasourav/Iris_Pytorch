# 🌸 Iris Flower Classification using Neural Network (PyTorch)

## 📑 Project Description

This is a simple **Neural Network (NN)** model built using **PyTorch** to classify the famous **Iris dataset** into its respective species:
- **Setosa**
- **Versicolor**
- **Virginica**

The model is trained and evaluated on the dataset, achieving high accuracy and good generalization.

---

## 💾 Dataset

- **Dataset Source**: [Iris Dataset](https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv)
- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Labels**:
  - Variety (species of Iris flower)

---

## ⚙️ Technologies & Libraries Used

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- scikit-learn (for train-test split)

---

## 🚀 Steps Performed

1. **Data Collection**: Collected Iris dataset from the web.
2. **Data Preprocessing**:
   - Encoded target labels (species names) to numerical format (0.0, 1.0, 2.0).
   - Split the dataset into **training and testing sets**.
   - Converted data into **PyTorch tensors** for model training.
3. **Model Building**:
   - Created a custom Neural Network using `torch.nn.Module`.
   - Network architecture:
     - Input layer: 4 neurons
     - Hidden layers: 2 hidden layers with 8 neurons each (ReLU activation)
     - Output layer: 3 neurons (for 3 classes)
4. **Training**:
   - Used **CrossEntropyLoss** as the loss function.
   - Optimized using **Adam optimizer**.
   - Trained for 100 epochs and plotted loss curve.
5. **Model Evaluation**:
   - Evaluated model on the test set.
   - Calculated **loss and accuracy**.
6. **Model Saving**:
   - Saved the trained model using `torch.save()` as `Iris_Model.pt`.
7. **Model Testing on New Data**:
   - Tested model on unseen/new data to predict flower variety.

---

## 📊 Observations

- The model achieved **100% accuracy** on the test set after training.
- Loss reduced significantly over epochs, demonstrating good learning.
- Model was able to correctly predict unseen data.
- **Effect of random seed and random_state**:
  - Observed that **higher values of random_state and manual_seed** sometimes caused **higher loss and reduced accuracy** due to the way the dataset was split and model initialized.
  - Lower values of seed and random_state often gave better splits and initializations, leading to **lower loss and higher accuracy**.
- Saved model (`.pt` file) can be reloaded and used for future predictions without retraining.

---

## ✅ How to Use the Saved Model

```python
# Load the saved model
new_model = Model()
new_model.load_state_dict(torch.load('Iris_Model.pt'))
new_model.eval()

# Predict on new data
new_data = torch.tensor([5.9, 3.0, 5.1, 1.8])  # Example data
with torch.no_grad():
    prediction = new_model(new_data)
    print(prediction)
