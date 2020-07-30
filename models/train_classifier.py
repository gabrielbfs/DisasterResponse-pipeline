import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'stopwords', 'wordnet'])


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """load the database data
    input:
        - database_filepath (str): path to database
    return:
        - X (arr): array with messages
        - Y (dataframe): dataframe with categories classification
        - Y.columns (list): category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disasters', con=engine)
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis='columns')
    
    return X, Y, Y.columns


def tokenize(text, lemmatizer=WordNetLemmatizer()):
    """tokenize and transform input text, used in Vectorizer class
    input:
        - text (str)
        - lemmatizer (model)
    return:
        - tokens
    """
    http_regex = r'http\S+'
    www_regex = r'www.\S+'
    text = re.sub(http_regex, 'url', text)
    text = re.sub(www_regex, 'url', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    text = re.sub(' +', ' ', text)
    
    # tokenization and lemmatization
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


def build_model():
    """
    input: None
    return:
        - Grid Search model with pipeline and Classifier
    """
    pipeline = Pipeline([
                        ('vectorizer',TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 1))),
                        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
                        ])

    parameters = {
                    'vectorizer__max_features': [50, 100, 250],
                    'clf__estimator__n_estimators': [100, 150]
                 }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """print model results
    input:
        - model (object): estimator-object
        - X_test (arr): X test array
        - y_test (arr): y test array
        - category_names (list): list of category strings
    return: None
    """
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        print('-'*50)
        print('{}: {}'.format(i, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
        

def save_model(model, model_filepath):
    """export model
    input:
        - model (object): estimator-object
        - model_filepath (str): path to export model
    return: None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()