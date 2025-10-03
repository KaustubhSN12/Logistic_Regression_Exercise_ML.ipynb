# 🩺 Logistic Regression – Diabetes Prediction

This repository demonstrates the implementation of a **Logistic Regression** model to predict diabetes using the **PIMA Indians Diabetes dataset**.  
It covers model training, prediction, evaluation (confusion matrix, accuracy, precision, recall), and visualization with **heatmaps** and **ROC curves**.

---

## 📌 Overview
- **Algorithm**: Logistic Regression (Supervised Learning – Classification)  
- **Dataset**: `diabetes.csv` (PIMA Indians Diabetes Dataset)  
- **Goal**: Predict whether a patient has **Diabetes (1)** or **No Diabetes (0)** using medical attributes.  

---

## 📊 Dataset Information
The dataset consists of the following attributes:

- `Pregnancies`  
- `Glucose`  
- `BloodPressure`  
- `SkinThickness`  
- `Insulin`  
- `BMI`  
- `DiabetesPedigreeFunction`  
- `Age`  
- `Outcome` → Target variable (0 = No Diabetes, 1 = Diabetes)  

**Example records**:  

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
|-------------|---------|---------------|---------------|---------|------|--------------------------|-----|---------|
| 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                    | 50  | 1       |
| 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                    | 31  | 0       |
| 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                    | 32  | 1       |

---

## ⚙️ Steps Covered

### 🔹 1. Data Loading & Splitting
```python
import pandas as pd
from sklearn.model_selection import train_test_split

pima = pd.read_csv("diabetes.csv")
feature_cols = ['Pregnancies','Insulin','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = pima[feature_cols]
y = pima.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
````

---

### 🔹 2. Training the Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
```

---

### 🔹 3. Predictions & Confusion Matrix

```python
y_pred = logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
```

📌 **Example Output**:

```
[[117  13]
 [ 24  38]]
```

---

### 🔹 4. Visualizing Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt="g")
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.show()
```

---

### 🔹 5. Evaluation Metrics

```python
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
```

📌 **Example Results**:

* **Accuracy**: 0.81
* **Precision**: 0.74
* **Recall**: 0.61

---

### 🔹 6. ROC Curve & AUC

```python
y_pred_proba = logreg.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr, tpr, label="AUC="+str(round(auc,2)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)
plt.show()
```

📌 Example AUC Score: **0.86**

---

## 🔑 Key Learning Outcomes

✔ Logistic Regression for **binary classification**
✔ Evaluate models with **confusion matrix, accuracy, precision, recall**
✔ Understand **ROC curve** & **AUC score**
✔ Learn the **trade-off between sensitivity and specificity**

---

## 📚 References

* [Scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [Understanding Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/)

---

## 🔗 Explore My Other Repositories

* 🥔 [Simple Linear Regression Exercise](https://github.com/KaustubhSN12/Simple-Linear-Regression-ML)
* ⚡ [SVM Algorithm Exercise](https://github.com/KaustubhSN12/SVM_Exercise_ML)
* 🌸 [KNN Algorithm Exercise](https://github.com/KaustubhSN12/KNN_Algorithm_Exercise_ML)
* 🤖 [Naive Bayes Algorithm Exercise](https://github.com/KaustubhSN12/Naive-bayes-algorithm_ML_Exercise)
* 🚀 [K-Means Clustering Exercise](https://github.com/KaustubhSN12/Kmeans_Cluster_Exercise_ML)

---

## 📜 License

This project is licensed under the **MIT License** – free to use and share with credit.

---

✨ *Star this repository if you found it useful for learning Logistic Regression!*

```
