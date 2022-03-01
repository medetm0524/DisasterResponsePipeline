import sys
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
from custom_transformer import StartingVerbExtractor

def load_data(database_filepath):
    """
    Load df from database specified in parameter and split it into X and Y dataframe as numpy values
    
    Parameter:
    database_filepath 
    
    Return
    X, Y, and columns names as list
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    X = df[['message']]
    X_np = X["message"].values
    
    Y = df[['related', 'request', 'offer','aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
            'security', 'military', 'child_alone', 'water', 'food', 'shelter',
            'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
            'infrastructure_related', 'transport', 'buildings', 'electricity',
            'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
            'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
            'other_weather', 'direct_report']]
    
    Y_np = Y[['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']].values
    Y_np = Y_np. astype(int)
    
    columns=['related', 'request', 'offer','aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    
    return X_np, Y_np, columns

def tokenize(text):
    """
    tokenize function cleans, tokenize and lemmatize each word in text message

    Parameter:
    test as message
    
    Return 
    clean text as list of tokens 
    """
    # Remove punctuation characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    #Split text into words using NLTK
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build_model is function with ML pipeline model
    function also contains parameters for GridSearchCV 
    
    Return
    cv model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
        
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])

    parameters = {
        'clf__estimator__n_neighbors':[9,10]
        }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate_model performs evaluation and prints classification report for each category

    Parameters:
    model is cv model 
    X_test is X test dataset
    y_test is Y test dataset
    category_names is list of categories 
    
    """
    y_pred = model.predict(X_test)
    
    y_test_df = pd.DataFrame(Y_test, columns=category_names)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)

    print("\nBest Parameters:", model.best_params_)

    for i in range(y_test_df.shape[1]):
        print("Test {} in dataframe \n".format(y_test_df.columns[i]),classification_report(y_test_df.iloc[:,i], y_pred_df.iloc[:,i]))

def save_model(model, model_filepath):
    """
    save model function that saves model as pickle file
    
    Parameters:
    ML model
    model_filepath where to save model
    
    Return
    """
    
    filename = '{}'.format(model_filepath)
    pickle.dump(model, open(filename, 'wb'))


def main():
    """
    main function that executes all functions above
    """
    
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
