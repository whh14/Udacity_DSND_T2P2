import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download(['punkt', 'stopwords', 'wordnet'] )


def load_data(database_filepath):
    """load data function.

    load the sqlite database as pd.Dataframe.

    Args:
        database_filepath: str, input file path.

    Returns:
        X: pd.DataFrame, feature variable
        Y: pd.DataFrame, target variable
        category_names: list of category names of target variables

    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)

    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """tokenize function.

    Apply tokenization to the text string for machine learning algorithm.

    Args:
        text: str, input string text to be tokenize.

    Returns:
        clean_tokens: list, list of token for machine learning algorithm

    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    tokens = [w for w in words if w not in stopwords.words("english")]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """build_model function.

    build machine learning pipeline with GridsearchCV.

    Args:
        None.

    Returns:
        cv: GridSearchCV model

    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 10000),
        'tfidf__use_idf': (True, False),
        # 'clf__n_estimators': [50, 100, 200],
        # 'clf__min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, cv=3, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate model function.

    Evaluate model model performacne on the test set

    Args:
        model: model to be evaluated
        X_test: feature variable of the testing set
        Y_test: target variable of the testing set
        category_names: category names of target variables

    Returns:
        None

    """

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """save model function.

    Save  model as a pickle file

    Args:
        model: model to be saved
        model_filepath: str, file path of the pickle file

    Returns:
        None

    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train.values)
        
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