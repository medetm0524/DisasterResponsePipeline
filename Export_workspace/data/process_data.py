# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:42:52 2022

@author: mmd20
"""

import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load_data function loads the data and merges two dataframes together based on "ID" column.
    Parameters:
    messages_filepath is the path for disaster_messages.csv file
    categories_filepath is the path for disaster_categories.csv file.
    
    Returns:
    merged dataframs df
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    clean_data function cleans df by splitting each category from category column
    
    Parameter:
    df database
    
    Return:
    clean database with expanded category column
    
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    
    category_colnames = []
    for i in range(len(row)):
        category_colnames.append(re.match(r"[a-zA-z]+",row[i]).group(0))
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].apply(lambda x: int(x[0]))
    
    df.drop(['categories'], axis=1, inplace=True)
    
    #Make sure all data is binary (0 or 1), if value is more than 1, convert it into 1
    for column in categories:
        categories[column] = categories[column].astype(int)
        categories[column] = categories[column].apply(lambda x: 1 if x>1 else x)
    
    df = pd.concat([df, categories], axis=1)
    
    return df

def save_data(df, database_filename):
    """
    save_data function saves df in specified database
    
    Parameters:
    df as clean dataframe
    
    database_filename is specified database path 
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('{}'.format(database_filename), engine, index=False, if_exists= 'replace') 


def main():
    """
    Run main function that executes all functions
    
    """
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
