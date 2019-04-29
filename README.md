# Udacity_DSND_T2P2
Udacity Data Scientist Nanodegree program, term 2
Project 3: Disaster Response Pipeline

## Project summary

The gola of this project is to build a machine learning model for analysing and classifying disaster messages with the dataset provided by Figure Eight.  There three main steps for this project.  The first step is to build a ETL (extract, transform, load) pipeline. Second step is to build a machine learning pipeline, and lastly a web app for depolying the machine learning model.  

## File Descriptions

1. data/process_data.py: ETL pipeline that load csv files, cleans data and save as SQLite database

2. models/train_classifier.py: ML pipeline that load the data provided by Figure Eight and construct a machine leering model and output the model as a pickle file

3. app/run.py: The web app to display information relates to the dataset and also classify message based on the trained model. 


## Instructions for running the Python scripts and web app 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Acknowledgements
I would like to thank Udacity for setting up the project and Figure Eight for providing the dataset.
