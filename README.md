# 🚦 Traffic Accident Prediction using CGNN

##  Overview
This project aims to predict traffic accident severity using a **Causal Graph Neural Network (CGNN)** approach.  
It combines **causal relationships between features** with deep learning to improve interpretability and performance.

---

##  Objectives
- Predict accident severity (multi-class classification)
- Incorporate causal relationships between variables
- Compare CGNN performance with standard models
- Enable explainability through causal structure

---

##  Methodology

### 🔹 1. Data Processing
- Data cleaning and preprocessing
- Feature engineering
- Encoding categorical variables
- Train/Validation/Test split

### 🔹 2. Causal Discovery
- Identify relationships between variables
- Construct causal graph using statistical methods

### 🔹 3. CGNN Model
- Treat features as nodes in a graph
- Perform message passing between nodes
- Learn feature interactions via graph structure

### 🔹 4. Training
- Loss: CrossEntropyLoss
- Optimizer: Adam
- GPU acceleration (Google Colab)

### 🔹 5. Evaluation
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

##  Project Structure

```

CGNN_Traffic_Accident/
│
├── configs/
│   └── phase4_config.yaml
│
├── data/
│   ├── causal_graphs/
│   ├── processed/
│   └── neural_models/
│
├── src/
│   ├── causal_discovery/
│   ├── data_processing/
│   └── neural_network/
│       ├── cgnn_model.py
│       ├── trainer.py
│       └── evaluator.py
│
├── results/
│   └── phase4/
│
├── main_phase4.py
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/CGNN_Traffic_Accident.git
cd CGNN_Traffic_Accident

pip install -r requirements.txt and then run the model on training data
```

---

## 📊 Sample Output

```
Epoch 1
Train Accuracy: 0.52
Validation Accuracy: 0.48
```

---

## 🧪 Results

| Metric      | Value                                    |
| ----------- | ---------------------------------------- |
| Accuracy    | ~35% (CGNN baseline)                     |
| F1 Score    | ~0.18                                    |
| Observation | Model struggles due to weak causal graph |

---

## ⚠️ Limitations

* Causal graph quality significantly affects performance
* Tabular datasets may not always benefit from GNNs
* Over-smoothing can occur with dense graphs
* Requires careful feature encoding

---

## 🔄 Future Improvements

* Improve causal graph construction
* Use hybrid models (CGNN + MLP)
* Hyperparameter tuning
* Better feature selection
* Use larger dataset

---

## 🧑‍💻 Author

**Rohan Roy Chowdhury**

* GitHub: https://github.com/Rohan-45-design

---

## 📌 Conclusion

This project demonstrates:

* The integration of causal reasoning with deep learning
* Practical challenges of applying CGNN to tabular data
* Importance of proper feature engineering and graph design

---

## ⭐ Acknowledgements

* PyTorch
* Scikit-learn
* Research papers on CGNN and causal inference

```
```
