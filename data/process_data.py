import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    """ load messages and categories dataframes from filepaths
    input:
        - messages_filepath (str) path to file
        - categories_filepath (str): path to file
    return:
        - df: pandas DataFrame with merged data
    """
    # load messages dataset and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(left=messages, right=categories, on=['id'], how='outer')
        
    return df


def clean_data(df):
    """clean dataframe with message and categories information (loaded df)
    input:
        - df: raw pandas DataFrame
    return:
        - df: processed pandas DataFrame
    """    
    # create a dataframe with 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    categories.columns = row.apply(lambda x: x[:-2])
    
    # convert category values to only numbers (0 or 1)
    for column in categories:
        # set each value to the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from df
    df = df.drop('categories', axis='columns')
    # concatenate the df with categories dataframe
    df = pd.concat([df, categories], axis='columns')
    # drop duplicates
    df = df.drop_duplicates()
    
    # verify there is no duplicated information
    assert df.duplicated().sum() == 0
    return df


def save_data(df, database_filename):
    """export dataframe to database
    input:
        - df: pandas DataFrame
        - database_filename
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disasters', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()