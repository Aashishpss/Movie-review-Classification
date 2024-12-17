# Movie Review Classification

This project aims to predict whether a movie review is positive or negative based on the text of the review. The dataset used for this project contains labeled reviews, and the goal is to train a model to classify the sentiment of unseen reviews. We use text classification techniques, particularly TF-IDF vectorization and a Linear Support Vector Classifier (SVC).
## Features

    Sentiment Analysis: Classifies movie reviews into positive or negative based on the review text.
    Pipeline Workflow: A machine learning pipeline that converts raw text into features and applies a classifier.
    Performance Metrics: Includes evaluation metrics like classification report and confusion matrix.

## Requirements

This project requires Python 3.x and the following libraries:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn

You can install these dependencies using pip:

pip install numpy pandas matplotlib seaborn scikit-learn

Installation

    Clone this repository to your local machine.

git clone https://github.com/yourusername/review-classification.git
cd review-classification

Install the required dependencies.

    pip install -r requirements.txt

    Download the dataset moviereviews.csv from Stanford Sentiment Dataset, and place it in the project directory.

## Usage

    Load the dataset using pandas to review the data.
    Preprocess the data by removing NaN values and blank strings.
    Split the data into training and test sets.
    Train a model using TF-IDF vectorization and a Linear SVC classifier.
    Evaluate the model performance using a classification report and confusion matrix.
    Use the trained model to classify new reviews.

Run the script with the following command:

    python review_classification.py

## Example

    review1 = "The action scenes of this movie are great, which makes it popular among the youths, resulting in a higher rating."
    print(pipe.predict([review1]))  # Output: ['pos']

## Code Explanation
1. Data Preprocessing

        The dataset is loaded into a pandas dataframe, and reviews with missing or blank text are removed.
        The labels (pos for positive and neg for negative) are counted to ensure balance in the dataset.

2. Train/Test Split

        The dataset is split into features (X) and labels (y), and further into training and testing sets using train_test_split from sklearn.model_selection.

3. Model Pipeline

        A pipeline is created using TfidfVectorizer (to convert text into numerical features) and LinearSVC (to classify the reviews).
        The model is trained on the training data using pipe.fit(X_train, y_train).

4. Model Evaluation

        After training the model, predictions are made on the test data, and the model's performance is evaluated using classification_report and a confusion matrix.

5. Classifying New Reviews

        After the model is trained, it can be used to predict the sentiment of new reviews.
