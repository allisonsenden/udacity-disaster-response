
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from nltk.corpus import stopwords
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
    Input:
        database_filepath or the path to SQLite db
    Output:
        X: feature DataFrame
        Y: label DataFrame
        category_names: 36 category labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterDf', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Input:
        text: original text for messages
    Output:
        clean_tokens: tokenized text for model
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    words = [w for w in words if w not in stopwords.words('english')]

    clean_tokens = []
    for w in words:
        clean_tok = lemmatizer.lemmatize(w).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Input: 
        none
    Output: 
        Builds an ML pipeline to process messages
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier()
        ))
    ])
    #print(pipeline.get_params().keys())
    parameters = {
        'clf__estimator__learning_rate': [0.1]
    }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, verbose=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
        model: pipeline built in build_model
        X_test: test features
        Y_test: test labels
        category_names: 36 category labels
    Output:
        classification report for 36 categories
    """
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), target_names=category_names))
    #print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Input:
        model: built model from build_model
        model_filepath: destination path to save .pkl file
    Output:
        saved model as pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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









# import sys


# def load_data(database_filepath):
#     pass


# def tokenize(text):
#     pass


# def build_model():
#     pass


# def evaluate_model(model, X_test, Y_test, category_names):
#     pass


# def save_model(model, model_filepath):
#     pass


# def main():
#     if len(sys.argv) == 3:
#         database_filepath, model_filepath = sys.argv[1:]
#         print('Loading data...\n    DATABASE: {}'.format(database_filepath))
#         X, Y, category_names = load_data(database_filepath)
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
#         print('Building model...')
#         model = build_model()
        
#         print('Training model...')
#         model.fit(X_train, Y_train)
        
#         print('Evaluating model...')
#         evaluate_model(model, X_test, Y_test, category_names)

#         print('Saving model...\n    MODEL: {}'.format(model_filepath))
#         save_model(model, model_filepath)

#         print('Trained model saved!')

#     else:
#         print('Please provide the filepath of the disaster messages database '\
#               'as the first argument and the filepath of the pickle file to '\
#               'save the model to as the second argument. \n\nExample: python '\
#               'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


# if __name__ == '__main__':
#     main()