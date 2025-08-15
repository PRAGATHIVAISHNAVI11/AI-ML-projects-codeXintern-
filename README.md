🧠 Machine Learning Projects Portfolio
Overview

This repository contains two beginner-to-intermediate Machine Learning projects demonstrating classification, data preprocessing, feature extraction, and model evaluation.
Both projects are fully implemented in Python using scikit-learn, pandas, Matplotlib, Seaborn, and NLTK.

1️⃣ Iris Flower Classification

Objective:
Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on petal and sepal measurements.

Key Features & Steps:

Loaded the Iris dataset from scikit-learn.

Explored the dataset visually with scatter plots, histograms, and pair plots.

Split data into training and test sets.

Trained and compared Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree classifiers.

Evaluated performance using accuracy, precision, recall, F1-score, and confusion matrices.

Visualized decision boundaries for petal features.

Compared model performance with accuracy bar charts.

Skills Gained:

Numeric data analysis & visualization

Classification modeling & algorithm comparison

Model evaluation & decision boundary visualization

Python, scikit-learn, Matplotlib, Seaborn

Results & Visuals:

All classifiers achieved 100% accuracy on the test set.

Petal length & width are the most important features.

Confusion Matrices:

Logistic Regression	
<img width="640" height="480" alt="confusion_matrix_Logistic_Regression" src="https://github.com/user-attachments/assets/7b31d283-d123-4403-b242-760103d1ce59" />

K-Nearest Neighbors	
<img width="640" height="480" alt="confusion_matrix_K-Nearest_Neighbors" src="https://github.com/user-attachments/assets/d3d6ef74-a7e9-4f5c-9361-f6477c996e0c" />

Decision Tree
<img width="640" height="480" alt="confusion_matrix_Decision_Tree" src="https://github.com/user-attachments/assets/8d2c074b-1261-467f-8788-d2bd9b5eb0b7" />

Accuracy Comparison:
<img width="600" height="400" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/f5db486b-8982-4ae4-a818-7ecc4cc337cf" />


Run Instructions:

cd Iris_Flower_Classification
python iris_classifier.py

View plots, confusion matrices, and accuracy charts.


2️⃣ Spam Mail Detector

Objective:
Classify messages as spam or ham (non-spam) using textual data.

Key Features & Steps:

Loaded the SMS Spam Collection dataset from UCI.

Applied text preprocessing: lowercasing, punctuation removal, stopwords removal, tokenization.

Converted text into numeric features with TF-IDF vectorization.

Split data into training and test sets.

Trained and compared Naive Bayes and Logistic Regression classifiers.

Evaluated performance using accuracy, precision, recall, F1-score, confusion matrices, and prediction confidence scores.

Developed an interactive demo to type a message, select a model, and get predictions with confidence bar chart visualization.

Skills Gained:

Text preprocessing & feature extraction

Basic NLP & classification modeling

Model evaluation & interactive prediction

Python, scikit-learn, Matplotlib, NLTK

Screenshots:
<img width="1920" height="1017" alt="Screenshot (854)" src="https://github.com/user-attachments/assets/399eff8b-1066-4b30-b2e5-35e14b9407d6" />

<img width="1920" height="1024" alt="Screenshot (856)" src="https://github.com/user-attachments/assets/4e27c06d-ff3d-4a99-a681-ba0e9457bfb2" />

Run Instructions:

cd Spam_Mail_Detector
python spam_mail_detector.py


Enter a message and select a model (Naive Bayes / Logistic Regression).

See prediction, confidence score, and confidence bar chart.

📦 Repository Structure
ML-Portfolio/
├─ Iris_Flower_Classification/
│   ├─ iris_classifier.py
│   ├─ confusion_matrix_Decision_Tree.png
│   ├─ confusion_matrix_K-Nearest_Neighbors.png
│   ├─ confusion_matrix_Logistic_Regression.png
│   ├─ accuracy_comparison.png
├─ Spam_Mail_Detector/
│   ├─ spam_mail_detector.py
├─ README.md
├─ requirements.txt

💻 Requirements

Install necessary Python packages:

pip install -r requirements.txt


requirements.txt

pandas
scikit-learn
matplotlib
seaborn
nltk
