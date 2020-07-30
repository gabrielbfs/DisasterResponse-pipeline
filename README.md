# Disaster Response Pipeline Project



### Project Overview:

In this Disaster Response Pipeline Project, we applied skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

The data set contain real messages that were sent during disaster events. The project consists of creating a machine learning pipeline to categorize these events so that the algorithm can send a messages to an appropriate disaster relief agency.



### Projects Components:

##### 1. ETL Pipeline

A Python script, `process_data.py`, a data cleaning pipeline that:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

##### 2. ML Pipeline

A Python script, `train_classifier.py`, a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

##### 3. Flask Web App

- Results in a Web App using Plotly



### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### VISUALIZATION:

![DisasterResponseProject_analyze-text](figures\DisasterResponseProject_analyze-text.PNG)



![DisasterResponseProject_genre-distribution](figures\DisasterResponseProject_genre-distribution.PNG)