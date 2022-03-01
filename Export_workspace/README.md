# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

###Project Motivation
Model helps to categorize incoming messages to an appropriate disaster relief agency.
This project is a part of Udacity, Data Science nanodegree journey. 

###File Description
In data folder, 2 source files:
	- disaster categories, which contains id of each messages and assign a category for it
	- disaster messages, which contains disaster messages 

In data folder, process_data.py script loads files, cleans, and saves exported dataframe as database

In models folder, train_classifier.py leads dataframe from database, perform NLP operations (tokenize), build pipeline, and perform evaluation for each categories.
And saves model using pickle. 

### Result
model KNeighborsClassifier classifier provides around 0.85 accuracy average over all categories. 
Try use another ML models to improve the accuracy. 
Ex. I need water message model categories to Relate, Request, Aid Related, Water and Food categories. Which is right!


### Licensing, Authors and Acknowledgements
Feel free to copy and use codes.
