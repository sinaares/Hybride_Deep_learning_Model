Credit Card Fraud Detection using SOM & ANN
This project uses a combination of Self-Organizing Maps (SOM) and an Artificial Neural Network (ANN) to detect potential fraud in credit card applications.

Overview
The idea is to first identify unusual patterns in the data using SOM, which is an unsupervised learning technique. Based on the SOM output, potential frauds are identified visually. These suspected frauds are then used to label the dataset and train a neural network (ANN) that can predict the likelihood of fraud for any customer.

Workflow:
Self-Organizing Map (SOM):

Used to identify unusual application behavior (outliers).

Visualizes a 2D grid where different color intensities represent the average distance from surrounding nodes (potential anomalies).

We manually select grid coordinates that likely indicate fraud.

Artificial Neural Network (ANN):

Uses the "fraud" labels generated from SOM as targets.

Learns to classify other customers based on their features.

Outputs probabilities of being a fraud.

Technologies Used
Python

Pandas, NumPy for data handling

MiniSom for SOM

Matplotlib for plotting

Scikit-learn for scaling

TensorFlow/Keras for the ANN

Dataset
The dataset used is Credit_Card_Applications.csv. It contains customer application data. Each row is a customer, and the last column is whether their application was approved (1) or rejected (0).

Make sure this file is in the same folder as the Python script.
