
# 📘 Lab 01 — Complete Solution (Parts 1–3)

This repository contains a full solution to **Lab Assignment #1** from the Machine Learning course.  
The lab consists of three major parts:

- **Part 1:** Matrix calculus & kNN
- **Part 2:** ML pipeline — preprocessing, PCA, logistic regression, ensembling
- **Part 3:** SVM & kernels on synthetic datasets

All solutions include:

- reproducible code
- clean visualizations
- detailed explanations
- final conclusions for each part

---

# 🧮 Part 1 — Matrix Calculus & kNN

### ✔ Topics:

- vector & matrix differentiation
- trace trick
- Frobenius norm gradients
- matrix factorization gradient
- L1- and L2-based nearest neighbors
- distance matrix interpretation
- classification behavior of k-NN

### ✔ Key Results:

- all gradients (∂J/∂A, ∂J/∂S) derived correctly
- 1-NN behavior explained via distance-matrix patterns
- effect of data preprocessing on L1 kNN analyzed
- theoretical questions answered cleanly and concisely

---

# 🔧 Part 2 — ML Pipeline: Preprocessing, PCA, Models

### ✔ Workflow implemented:

- train/test split
- feature scaling
- PCA: explained variance → dimension reduction
- logistic regression (raw & PCA-transformed)
- decision tree with CV
- bagging (LR & DT ensembles)
- random forest
- XGBoost (native API, no sklearn wrapper)
- learning curves

### ✔ Highlights:

- tuned PCA (≈95% variance → 8 components)
- multinomial LR + confusion matrix
- DT optimal depth via CV
- bagging improves stability
- RF & boosting outperform single models
- full analysis of how ensemble size affects metrics
- learning curves demonstrate bias/variance behavior

---

# 🎯 Part 3 — SVM & Kernels

### ✔ Experiments:

- linear LR vs linear SVM
- SVM with polynomial, RBF, sigmoid kernels
- visualization of decision regions
- comparison of nonlinear boundaries
- PolynomialFeatures + Logistic Regression
- complex dataset (circles + moons, 4 classes)
- refactored grid search for LR+Poly and SVM RBF
- explicit vs implicit polynomial mapping explained

### ✔ Best Models:

- **RBF SVM:** 0.99 accuracy on moons
- **LR + Poly (deg=4):** 0.94 accuracy on complex dataset
- **RBF SVM:** 0.95 on complex dataset (best overall)

---

# 📝 Final Conclusions

### ✔ Part 1

Matrix calculus builds the foundation for gradient-based ML methods.  
k-NN behavior strongly depends on geometry of the dataset and distance preprocessing.

### ✔ Part 2

PCA dramatically simplifies the feature space without major loss of information.  
Logistic regression is strong on well-scaled data; tree-based models capture nonlinearities.  
Bagging, Random Forest, and Boosting consistently outperform single models.

### ✔ Part 3

Linear models fail on nonlinear geometry, but both **Kernel SVM** and **PolynomialFeatures+LR**  
solve the moons and circles datasets extremely well.  
RBF kernels remain the most flexible and robust across nonlinear structures.
