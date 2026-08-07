# 🛡️ Fake Job Posting Detector

An AI-powered dashboard that detects fraudulent job postings using Machine Learning and Natural Language Processing, helping job seekers stay safe from employment scams.

## 🎯 Problem

Online job scams are a growing threat — fraudulent postings promise unrealistic salaries, request upfront payments, or use urgency tactics to trick job seekers. This project uses ML to automatically flag suspicious job listings before applicants fall victim to them.

## ✨ Features

- **Real-time job analysis** — paste any job title/description and get an instant Fake/Safe prediction with a confidence score
- **Explainable AI** — highlights the specific "red flag" words that push a listing toward Fake, and "trust" words that support Safe, so the reasoning isn't a black box
- **Interactive dashboard** — live stats (jobs checked, fake vs. safe counts), scan history, and a recent-scans table
- **Built-in safety tips** — common patterns to watch for when job hunting

## 🧠 How It Works

1. **Data**: Trained on ~18,000 real job postings (Kaggle's *Real/Fake Job Posting Prediction* dataset)
2. **Text processing**: Job title + description converted to numerical features using **TF-IDF vectorization**
3. **Model**: **Logistic Regression** classifier with class balancing to handle the real-world imbalance between genuine and fraudulent postings
4. **Explainability**: Model coefficients are used to surface which words most influenced each prediction

**Performance:** ~95% overall accuracy, with 80% recall on fraudulent postings (prioritizing catching scams over minimizing false alarms).

## 🛠️ Tech Stack

- **Python**
- **scikit-learn** — TF-IDF vectorization, Logistic Regression
- **pandas / NumPy** — data processing
- **Streamlit** — interactive web dashboard
- **Plotly** — explainability visualizations

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/guptasanchita1998-sys/fake-job-detector.git
cd fake-job-detector

# Install dependencies
pip install streamlit pandas scikit-learn nltk joblib plotly

# Download the dataset from Kaggle ("Real/Fake Job Posting Prediction")
# and place fake_job_postings.csv in this folder

# Train the model (creates fake_job_model.pkl and vectorizer.pkl)
python train_model.py

# Launch the dashboard
streamlit run app.py
```

## 📊 Dataset

[Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) — ~18,000 job postings labeled as real or fraudulent.

## 📌 Future Improvements

- Add support for checking a job posting directly from a URL
- Expand training data with India-specific scam examples
- Experiment with additional models (Random Forest, Naive Bayes) for comparison

---

*Built as a personal project to explore applied NLP and explainable AI.*
