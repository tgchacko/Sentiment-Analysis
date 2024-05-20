# Sentiment Analysis of Reviews

## Table of Contents

[Project Overview](#project-overview)

[Data Sources](#data-sources)

[Data Description](#data-description)

[Tools](#tools)

[EDA Steps](#eda-steps)

[Data Preprocessing Steps and Inspiration](#data-preprocessing-steps-and-inspiration)

[Graphs/Visualizations](#graphs-visualizations)

[Choosing the Algorithm for the Project](#choosing-the-algorithm-for-the-best-project)

[Assumptions](#assumptions)

[Model Evaluation Metrics](#model-evaluation-metrics)

[Results](#results)

[Recommendations](#recommendations)

[Limitations](#limitations)

[Future Possibilities of the Project](#future-possibilities-of-the-project)

[References](#references)

### Project Overview

The primary goal of this project is to effectively analyze customer reviews to understand the sentiment and quality perception of products based on user-generated content. The analysis aims to identify patterns and trends in the data that provide insights into customer satisfaction and product quality. Additionally, the project seeks to classify each review based on the sentiment expressed for each product, aiding in the qualitative assessment of feedback.

### Data Sources

The primary dataset used for this analysis contains detailed information about product reviews, including text reviews, ratings, and other metadata.

[Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

### Data Description

The dataset consists of 5,68,411 rows and 10 columns, including unique identifiers for reviews, products, and users, as well as textual data for reviews and summaries. The columns are:

1) Id: Unique identifier for each review.
2) ProductId: Unique identifier for the product being reviewed.
3) ProfileName: Name of the user profile.
4) HelpfulnessNumerator: Number of users who found the review helpful.
5) HelpfulnessDenominator: Number of users who indicated whether they found the review helpful or not.
6) Score: Rating given to the product by the reviewer.
7) Time: Timestamp when the review was posted.
8) Unemployment: The unemployment rate in the region
9) Summary: Summary of the review.
10) Text: Full text of the review.

![Data Description1](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/8c10ec6b-4081-4c90-9cfc-696505c59965)

![Data Description2](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/98ea23d5-0312-483e-b7f5-d63eec9ec14d)

### Tools

- Python: Data Cleaning and Analysis

    [Download Python](https://www.python.org/downloads/)

- Jupyter Notebook: For interactive data analysis and visualization

    [Install Jupyter](https://jupyter.org/install)
 
**Libraries**

Below are the links for details and commands (if required) to install the necessary Python packages:
- **pandas**: Go to [Pandas Installation](https://pypi.org/project/pandas/) or use command: `pip install pandas`
- **numpy**: Go to [NumPy Installation](https://pypi.org/project/numpy/) or use command: `pip install numpy`
- **matplotlib**: Go to [Matplotlib Installation](https://pypi.org/project/matplotlib/) or use command: `pip install matplotlib`
- **seaborn**: Go to [Seaborn Installation](https://pypi.org/project/seaborn/) or use command: `pip install seaborn`
- **scikit-learn**: Go to [Scikit-Learn Installation](https://pypi.org/project/scikit-learn/) or use command: `pip install scikit-learn`
- **NLTK**: Go to [NLTK Installation](https://pypi.org/project/nltk/) or use command: `pip install nltk`

### EDA Steps

Exploratory Data Analysis (EDA) involved exploring the reviews data to answer key questions, such as:

1) What is the distribution of scores?
2) How do review lengths vary?
3) What are the common themes in positive and negative reviews?

### Data Preprocessing Steps and Inspiration

1. #### Data Cleaning:
**Handling Missing Values**: Any missing values are identified and removed to ensure the quality of the data.
**Removing Duplicates**: Duplicate entries are checked and removed to ensure the uniqueness of data points for accurate analysis.
**Consistency Checks**: Ensuring that helpfulness numerators do not exceed denominators and standardizing text data for uniformity.

2. #### Data Transformation:
**Converting Data Types**: The data type for the column 'Time' is changed to 'datetime' from 'int64'.
**Feature Engineering**: New features such as helpfulness ratio, text length, and summary length are generated.

3. #### Text Preprocessing

1) **Tokenization**: Breaking down the text into individual words or tokens.
2) **Stop Words Removal**: Eliminating common words that offer little value for analysis.
3) **Lemmatization**: Converting words into their base form.
4) **Vectorization**: Transforming text data into numerical format using techniques like Count Vectorization and TF-IDF.

#### Inspiration for Data Preprocessing

The inspiration for the specific preprocessing steps comes from typical challenges encountered in natural language processing and sentiment analysis tasks, particularly noise reduction, dimensionality reduction, and bias removal.

### Graphs/Visualizations

![Distribution of Text Length](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/cb66e43a-d500-400d-89fb-4df3a0e57426)

![Distribution of Summary Length](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/c0e2a016-f5dd-4d88-993a-220ae573c7ec)

![Distribution of Review Scores](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/e66f9b82-cb0b-4108-870b-08b654ce5986)

![Distribution of Helpfulness Ratio](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/8e7dca40-1b29-46da-98b3-3814bf70e316)

![Scatter plot of Demand vs Rating(Score)](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/fc56c7f7-c88e-4e30-b8f0-92ff2139e456)

![Word Cloud - Good Reviews](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/b184e8da-454c-440c-8a3e-60fafe2096f6)

![Word Cloud - Bad Reviews](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/86ff2ec5-70c4-4f43-b76e-8ee77d54c0ac)

### Choosing the Algorithm for the Project

1) **Logistic Regression - TFIDF**: Uses term frequency-inverse document frequency (TFIDF) to weigh words based on their importance and logistic regression for binary classification, providing a balance of interpretability and performance.

2) **Naive Bayes - Count Vectorizer**: Utilizes the count vectorizer to transform text data into token counts and applies Naive Bayes for probabilistic classification, effective for large datasets and capturing word frequency.

3) **Logistic Regression - Count Vectorizer**: Combines count vectorizer for token counts with logistic regression to predict sentiment, suitable for linear relationships and high-dimensional data.

4) **Naive Bayes - TFIDF**: Employs TFIDF to emphasize important words and Naive Bayes for classification, balancing word importance and probabilistic predictions.

5) **NLTK SIA Polarity Scores**: Utilizes the Sentiment Intensity Analyzer from NLTK to quickly assess sentiment polarity scores, offering a simple and fast sentiment analysis approach.
  
### Assumptions

1) **Independence of Features**: Assuming that words are independent of each other.
2) **Linear Relationships**: Assuming linear separability of sentiment based on word presence.
3) **Text Preprocessing Decisions**: Assuming preprocessing steps adequately capture important features.
4) **Quality and Completeness of Data**: Assuming the dataset accurately represents the population of interest.
5) **Sentiment Labeling Accuracy**: Assuming sentiment labels are correct.

### Model Evaluation Metrics

1. **Accuracy**: Measures the proportion of total predictions that were correct.
2. **Precision**: Measures the accuracy of positive predictions.
3. **Recall(Sensitivity)**: Measures the ability to find all relevant cases within a dataset.
4. **F1 Score**: The harmonic mean of precision and recall.

### Results 

#### Breakdown of Each Model's Performance

1) **Logistic Regression - TFIDF**: Accuracy: 91.29%, Precision: 0.84 (negative), 0.93 (positive), Recall: 0.73 (negative), 0.96 (positive), F1-Score: 0.79 (negative), 0.95 (positive)

![Logistic Regression with TF-IDF Results](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/739e25ec-8a82-44bb-8177-222d16a2ee27)

2) **Naive Bayes - Count Vectorizer**: Accuracy: 89.42%, Precision: 0.77 (negative), 0.93 (positive), Recall: 0.73 (negative), 0.94 (positive), F1-Score: 0.75 (negative), 0.93 (positive)

![Naive Bayes Classifier with Count Vectorizer Results](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/37acfe65-bfc8-4086-8f73-0c3062630668)

3) **Logistic Regression - Count Vectorizer**: Accuracy: 91.64%, Precision: 0.84 (negative), 0.93 (positive), Recall: 0.76 (negative), 0.96 (positive), F1-Score: 0.80 (negative), 0.95 (positive)

![Logistic Regression with Count Vectorizer Results](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/e3c2a3d6-a05f-4c73-9030-77e550bce8c5)

4) **Naive Bayes - TFIDF: Accuracy**: 85.38%, Precision: 0.90 (negative), 0.85 (positive), Recall: 0.37 (negative), 0.99 (positive), F1-Score: 0.52 (negative), 0.91 (positive)

![Naive Bayes Classifier with TF-IDF Results](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/5163bb36-0863-450f-8028-f96d9d92573f)

5) **NLTK SIA Polarity Scores**: Accuracy: 81.97%, Precision: 0.74 (negative), 0.83 (positive), Recall: 0.26 (negative), 0.97 (positive), F1-Score: 0.39 (negative), 0.89 (positive)

![NLTK SIA Polarity Scores](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/137dc5b0-60bb-4935-91d0-b92f44acd903)

![Models Accuracy](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/cf7a4832-970a-4705-94e3-47b24b3eec17)

![Count of Products in Each Segment](https://github.com/tgchacko/Sentiment-Analysis/assets/169921893/0a0a65f5-8d9e-469c-831a-02d0fe98f78f)

| Model | Accuracy | Precision (Negative) | Precision (Positive) | Recall (Negative) | Recall (Positive) | F1-Score (Negative) | F1-Score (Positive) |
|-----------------------------------|----------|------|------|------|------|------|------|
| Logistic Regression - TF-IDF | 91.29% | 0.84 | 0.93 | 0.73 | 0.96 | 0.79 | 0.95 |
| Naive Bayes - Count Vectorizer | 89.42% |	0.77 | 0.93 | 0.73 | 0.94 |	0.75 | 0.93 |
| Logistic Regression - Count Vectorizer | 91.64% |	0.84 | 0.93 | 0.76 | 0.96 |	0.80 | 0.95 |
| Naive Bayes - TF-IDF | 85.38% | 0.90 | 0.85 | 0.37 | 0.99 | 0.52 | 0.91|

**Balanced Performance**: Logistic Regression – Count Vectorizer stands out as the best model due to its high accuracy and balanced precision and recall across both classes.

### Recommendations

1) Implement targeted improvements based on feedback from reviews.
2) Use sentiment analysis results to guide product development and marketing strategies.
3) Continuously update and refine the models with new data for improved accuracy.

### Limitations

1) Data Quality: Potential inaccuracies due to underreporting or subjective nature of reviews.
2) Model Limitations: Models may not capture all nuances of sentiment in reviews.
3) External Factors: Other factors not included in the analysis can impact sentiment.

### Future Possibilities of the Project
1) Advanced Predictive Modeling: Explore advanced models like NBEATS, NHITS, PatchTST, VARMAX, VAR, and KATS for enhanced accuracy.
2) Store-Specific/Product-Specific Analysis: Conduct detailed analysis for each product category in each store to uncover unique patterns and optimize models for individual characteristics.
3) External Factors Integration: Incorporate additional factors like economic indicators, social events, and regional factors for a comprehensive approach.

### References

1) Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O’Reilly Media, Inc.
2) Jurafsky, D., & Martin, J. H. (2019). Speech and Language Processing (3rd ed.).
