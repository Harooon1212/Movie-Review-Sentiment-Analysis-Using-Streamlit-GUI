# Movie Review Sentiment Analysis 🎬

This repository contains a Natural Language Processing (NLP) project developed for the NLP course at **Bahria University**.The project focuses on classifying movie reviews as **Positive** or **Negative** using machine learning and provides an interactive GUI for testing.

## 📋 Project Highlights
* **Balanced Dataset**: To address severe class imbalance in the initial data, the IMDB Movie Review dataset was integrated to provide a more robust training set.
* **Text Preprocessing**: Implements a comprehensive cleaning pipeline including HTML tag removal, punctuation stripping, lowercasing, and tokenization.
* **Model Training**: Utilizes a TF-IDF Vectorizer and Multinomial Naive Bayes (or similar) to achieve sentiment classification.
* **Interactive GUI**: A Streamlit-based web interface allows users to input custom reviews and see real-time sentiment predictions.

## 🛠️ Tech Stack
* **Language**: Python
* **Libraries**: `pandas`, `nltk`, `scikit-learn`, `seaborn`, `matplotlib`
* **Deployment**: Streamlit

## 🚀 How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/movie-review-sentiment-analysis.git
   ```
2. **Install dependencies**:
   
```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit App**:
   
```bash
   streamlit run app.py
   ```



## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
```
