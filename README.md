# Disaster Response Pipeline Project

### Installation:
Outside of Anaconda, there were 3 packages necessary to install: 'punkt', 'wordnet', 'stopwords'. This is handled in the scripts written.

### File Descriptions:
data/
	process_data.py - script to read and clean the messages to output to cleaned databse
    disaster_messages.csv - raw messages csv file
    disaster_categories.csv - raw category csv file
    DisasterResponse.db - database created with cleaned disaster messages dataset
models/
	train_classifier.py - script to build, train, and evaulate the model
    classifier.pkl - trained classifier outputted by pipeline model 
app/
	run.py - this is the flask file to run the app
    templates/ - contains html file for the web app viewing

### Results:
1. An ETL pipeline to handle the initial datasets. It reads in the data, cleans it and saves it to a SQLite database.
2. An ML pipeline to train a classifier with MultiOutputClassifier for 36 category labels on the disaster recovery messages.
3. A flask web application for easier viewing of the data and classifications results.

### Author:
Allison Senden

### Acknowledgements:
1. Udacity Data Scientist for Enterprise Nanodegree program
2. FigureEight for providing us all the data in order to complete this project.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3002/
