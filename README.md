
# **Human Emotion Classification using Machine Learning and Deep Learning**

This project is a comprehensive analysis and implementation of emotion detection using text. It explores both traditional machine learning algorithms and modern deep learning techniques to classify text into six human emotions: Joy, Fear, Anger, Love, Sadness, and Surprise.

---

## **Table of Contents**

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Modeling Approaches](#modeling-approaches)  
   - [Machine Learning Models](#machine-learning-models)
   - [Deep Learning with LSTM](#deep-learning-with-lstm)  
6. [Results and Metrics](#results-and-metrics)  
7. [Conclusion](#conclusion)

---

## **Overview**

The goal of this project is to classify text into six emotional categories:
- **Joy**
- **Fear**
- **Anger**
- **Love**
- **Sadness**
- **Surprise**

The project involves:
- Data cleaning and preprocessing
- Feature extraction using TF-IDF and one-hot encoding
- Implementing machine learning models
- Building and training an LSTM-based deep learning model
- Comparing the performance of all approaches

---

## **Dataset**

The dataset consists of text labeled with one of six emotions. The data is split into training and testing sets.

- **Shape**: 16,000+ rows with two columns: `Comment` and `Emotion`.
- **Sample Rows**:

| Comment                                    | Emotion  |
|--------------------------------------------|----------|
| I feel great today!                        | Joy      |
| This is an outrage!                        | Anger    |
| I’m feeling lonely and sad.               | Sadness  |

---

## **Exploratory Data Analysis (EDA)**

### Distribution of Emotions
A bar chart displays the distribution of emotions in the dataset:
- Joy and Sadness are the most frequent emotions.
- Surprise and Love are less common.

### Word Clouds for Emotions
Generated word clouds highlight common words associated with each emotion.

---

## **Data Preprocessing**

### Machine Learning Preprocessing:
1. **Text Cleaning**: Removed special characters, converted text to lowercase, removed stopwords, and applied stemming.
2. **Vectorization**: Used TF-IDF to convert text into numerical features.

### Deep Learning Preprocessing:
1. **Text Cleaning**: Similar to the machine learning preprocessing pipeline.
2. **One-Hot Encoding**: Converted words into numerical indices using a vocabulary size of 11,000.
3. **Padding**: Ensured all sequences are of the same length (300 words).

---

## **Modeling Approaches**

### Machine Learning Models
Four classifiers were evaluated using TF-IDF features:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Random Forest Classifier
4. Support Vector Machine (SVM)

**Pipeline**:
- Clean text
- Vectorize using TF-IDF
- Train and evaluate classifiers

---

### Deep Learning with LSTM
The LSTM model architecture:
- **Embedding Layer**: Converts words into dense vectors of size 150.
- **LSTM Layer**: Captures sequential dependencies with 128 units.
- **Dense Layers**: Adds two fully connected layers with dropout regularization.
- **Output Layer**: Predicts probabilities for six emotions using softmax.

**Hyperparameters**:
- Vocabulary Size: 11,000
- Sequence Length: 300
- Batch Size: 64
- Epochs: 15
- Optimizer: Adam

---

## **Results and Metrics**

### Machine Learning Results (TF-IDF)

| Model                        | Accuracy | Precision | Recall | F1-Score |
|------------------------------|----------|-----------|--------|----------|
| Multinomial Naive Bayes      | 0.65     | 0.85      | 0.31   | 0.46     |
| Logistic Regression          | 0.82     | 0.83      | 0.83   | 0.82     |
| Random Forest Classifier     | 0.85     | 0.85      | 0.85   | 0.85     |
| Support Vector Machine (SVM) | 0.81     | 0.82      | 0.82   | 0.81     |

Logistic Regression performed the best among the machine learning models.

---

### Deep Learning Results (LSTM)

| Metric         | Value   |
|----------------|---------|
| Accuracy       | *0.98*  |
| Loss           | 0.04    |


The LSTM model outperformed all machine learning models, achieving a significantly higher accuracy of **91%**.

---

## **Conclusion**

This project demonstrates the effectiveness of both machine learning and deep learning approaches for emotion detection. While logistic regression performed well among traditional classifiers, the LSTM-based deep learning model outshined all machine learning approaches by capturing sequential dependencies in the text.

### Key Highlights:
1. **Machine Learning**: Logistic Regression achieved an accuracy of 84% using TF-IDF features.
2. **Deep Learning**: The LSTM model reached an accuracy of 91%, making it the best-performing model for this task.
3. **Final Decision**: The LSTM-based deep learning model is recommended for production deployment due to its superior performance.

---

## **Future Scope**
1. Fine-tuning hyperparameters for even better performance.
2. Exploring transformer-based models like BERT or GPT for advanced results.
3. Extending the application to real-time emotion detection in chat systems.

---

