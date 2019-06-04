
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
 
def load_data(messages_filepath, categories_filepath):
    """
    Input:
        messages_filepath: path to messages csv file
        categories_filepath: path to categories csv file
    Output:
        df: Loaded data as pandas df
    """
    #read in csv
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merge
    df = pd.merge(messages,categories,on='id')
    return df 

def clean_data(df):
    """
    Input:
        merged df of messages and categories
    Output:
        df: clean pandas df
    """
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df.drop('categories',axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Input:
        Clean pandas df
        database_filename, database file(.db) destination path
    Output:
        saved SQLite database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterDf', engine, index=False)
    pass  


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