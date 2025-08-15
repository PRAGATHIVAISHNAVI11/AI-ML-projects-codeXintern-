# spam_mail_interactive_visual.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import string

# -------------------------
# 1️⃣ Load dataset
# -------------------------
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# -------------------------
# 2️⃣ Text preprocessing
# -------------------------
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(preprocess_text)

# -------------------------
# 3️⃣ Feature extraction
# -------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label']

# -------------------------
# 4️⃣ Split data and train models
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

for name, model in models.items():
    model.fit(X_train, y_train)

# -------------------------
# 5️⃣ Interactive prediction with confidence bar
# -------------------------
def predict_message(msg, model_name="Naive Bayes"):
    if model_name not in models:
        print(f"Model '{model_name}' not found. Using Naive Bayes.")
        model_name = "Naive Bayes"
        
    model = models[model_name]
    msg_clean = preprocess_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    
    prediction = model.predict(msg_vec)[0]
    probs = model.predict_proba(msg_vec)[0] * 100  # Convert to %
    
    labels = model.classes_
    
    print(f"\nModel: {model_name}")
    print(f"Message: {msg}")
    print(f"Prediction: {prediction.upper()} – {max(probs):.1f}% confident")
    
    # -------------------------
    # Plot confidence bar chart
    # -------------------------
    plt.figure(figsize=(5,3))
    plt.bar(labels, probs, color=['green','red'])
    plt.ylim(0,100)
    plt.title(f"{model_name} Prediction Confidence")
    plt.ylabel("Confidence (%)")
    for i, v in enumerate(probs):
        plt.text(i, v+1, f"{v:.1f}%", ha='center')
    plt.show()

# -------------------------
# 6️⃣ Demo loop
# -------------------------
print("=== Advanced Spam Mail Detector with Confidence Visualization ===")
print("Available models: Naive Bayes, Logistic Regression")

while True:
    user_input = input("\nEnter a message (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    
    chosen_model = input("Choose model (Naive Bayes / Logistic Regression): ")
    predict_message(user_input, chosen_model)
