---
title: "Timepass Trainer Demo"
author: "Sarthak"
format: revealjs
embed-resources: true
---

# Introduction

- This project demonstrates training a logistic regression model  
- Uses SGDClassifier with partial fitting on synthetic data  
- Shows progress bar and evaluation metrics with rich library

# Dataset Details

- Synthetic classification dataset with 1000 samples  
- 20 features with 15 informative features  
- Random dataset generation seed every run  

# Model Training

- Logistic Regression model trained over 50 epochs  
- Partial fit applied each epoch  
- Learning rate fixed at 0.01  
- Random model seed every run  

# Training Visualization

- Rich progress bar displays epoch progress  
- Logs accuracy every 10 epochs  
- Training completes with final performance metrics

# Metrics Reported

- Accuracy  
- Log Loss  
- Precision  
- Recall  
- Number of samples and features

# Summary

- Demonstrates incremental learning  
- Interactive terminal visualization with Rich  
- Easily extendable for real datasets and different models  

# Code Snippet Example
```bash
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="constant", eta0=0.01)

```
undefined