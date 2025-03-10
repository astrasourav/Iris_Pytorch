# Iris Flower Classification using Neural Network (PyTorch)

## 📊 Dataset
This project uses the classic **Iris flower dataset** with 150 samples divided into three species:
- Setosa
- Versicolor
- Virginica

Each sample contains 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

---

## 🚀 Model
A simple feed-forward neural network with the following architecture:
- Input Layer: 4 neurons
- Hidden Layer 1: 8 neurons (ReLU activation)
- Hidden Layer 2: 8 neurons (ReLU activation)
- Output Layer: 3 neurons (for 3 classes)

---

## 📈 Training Results
- **Loss decreases steadily over 100 epochs.**
- Achieved **100% accuracy** on the test set (Note: This may vary based on `random_state` and seed).
- Loss graph (`loss.png`) is included in the repository.

---

## ⚙️ Files Overview
| File              | Purpose                                 |
|-------------------|-----------------------------------------|
| `model.py`        | Model class definition                   |
| `train.py`        | Training code (optional to retrain)      |
| `predict.py`      | Load and predict using trained model     |
| `Iris_Model.pt`   | Saved PyTorch model                      |
| `loss.png`        | Training loss graph                      |
| `requirements.txt`| List of required Python libraries        |

---

## 📦 How to Use

### 1. Clone the Repository
```bash
git clone <https://github.com/astrasourav/Iris_Pytorch.git>
cd Iris-Flower-Classification


---

## ✅ Now What to Upload on GitHub:
- `model.py`
- `train.py`
- `predict.py`
- `Iris_Model.pt`
- `loss.png`
- `README.md`
- `requirements.txt`

---

## 🔑 Final Note:
- ✅ **Training code should be there but not auto-run (inside `if __name__ == "__main__"`)**.
- ✅ `.pt` model is uploaded for easy reuse.
- ✅ Prediction script works directly without needing to retrain.

---

If you want, I can **zip** this as a package and share, but you can create these files easily now.  
Let me know if you want that zip file or if you want to proceed! 🚀

