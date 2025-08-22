# Sentiment-Analysis-on-Restaurant-Reviews-Using-ML
"This project performs sentiment analysis on restaurant reviews using Machine Learning. Reviews are preprocessed with NLP techniques like stemming and stopword removal, converted into vectors, and classified using a Random Forest model to predict positive or negative sentiments."

# 🍴 Sentiment Analysis on Restaurant Reviews using ML  

## 📖 Project Overview  
This project focuses on analyzing customer restaurant reviews to determine whether they express **positive** or **negative** sentiment. By applying **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques, the system helps businesses gain insights into customer satisfaction and make data-driven improvements.  

## 🎯 Objectives  
- Preprocess restaurant reviews using NLP techniques.  
- Convert text data into numerical features using **Bag of Words / Count Vectorizer**.  
- Train a **Random Forest Classifier** for sentiment classification.  
- Evaluate the model's accuracy and performance.  
- Provide an easy-to-use solution for sentiment prediction.  

## 🛠️ Technologies Used  
- **Python**  
- **Google Colab / Jupyter Notebook**  
- **Scikit-learn** (Machine Learning)  
- **NLTK** (Natural Language Processing)  
- **Pandas & NumPy** (Data Handling)  
- **Matplotlib / Seaborn** (Visualization)  

## 📂 Project Structure  
📦 sentiment-analysis-restaurant-reviews
┣ 📜 Sentiment_Analysis_on_Restaurant_Reviews.ipynb # Main notebook with code
┣ 📜 Restaurant_review_model.pkl # Trained ML model
┣ 📜 count_v_res.pkl # CountVectorizer file
┣ 📜 README.md # Project documentation
┗ 📜 requirements.txt # Required dependencies



## ⚙️ How It Works  
1. **Text Preprocessing**  
   - Remove punctuation, stopwords, and special characters.  
   - Apply **stemming/lemmatization** to normalize words.  

2. **Feature Extraction**  
   - Convert reviews into numerical vectors using **CountVectorizer**.  

3. **Model Training**  
   - Train a **Random Forest Classifier** to classify sentiments.  

4. **Prediction**  
   - Input: A restaurant review (text).  
   - Output: Sentiment → **Positive** or **Negative**.  

## 🚀 How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/gaddam-omkar/Sentiment-Analysis-on-Restaurant-Reviews-Using-ML.git

2. Open the notebook in Google Colab or Jupyter Notebook.

3. Install dependencies:
pip install -r requirements.txt

4. Run all cells to train and test the model.


📊 Results

Achieved high accuracy using Random Forest Classifier.

Provides reliable sentiment classification for real-world reviews.

📌 Future Improvements

Build a Streamlit / Flask Web App for user-friendly interaction.

Expand dataset for better generalization.

Try advanced models like LSTMs or Transformers (BERT).

🙌 Acknowledgements

Dataset: Restaurant Reviews Dataset (Kaggle).

Libraries: NLTK, Scikit-learn, Pandas, NumPy.
