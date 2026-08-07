import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset load karo
df = pd.read_csv('fake_job_postings.csv')

# Missing values ko empty string se fill karo (description, title me)
df['description'] = df['description'].fillna('')
df['title'] = df['title'].fillna('')

# Title + description ko combine karke ek text column banao
df['text'] = df['title'] + ' ' + df['description']

# Features (X) aur Target (y) alag karo
X = df['text']
y = df['fraudulent']

# Train aur Test data me split karo (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Text ko numbers me convert karo (TF-IDF technique se)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model banao aur train karo
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Predictions karo test data pe
y_pred = model.predict(X_test_vec)

# Accuracy check karo
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nDetailed Report:\n", classification_report(y_test, y_pred))



# Custom job posting test karne ke liye function
def predict_job(title, description):
    text = title + ' ' + description
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    result = "FAKE" if prediction == 1 else "REAL"
    confidence = probability[prediction] * 100
    
    print(f"\nPrediction: {result}")
    print(f"Confidence: {confidence:.2f}%")

# Test karo ek example job ke sath
predict_job(
    "Work From Home - Earn $5000/week!",
    "No experience needed! Just pay a small registration fee of $50 to get started. Guaranteed income, easy money, urgent hiring!"
)

