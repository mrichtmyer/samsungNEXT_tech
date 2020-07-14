# Samsung NEXT Tech Interview
Matthew Richtmyer
July 14, 2020

# Overview:
* This repo documents a custom name entity recognition algorithm using a Naive Bayes classifier to predict NER classes for new words. 
* Two jupyter notebooks document data cleaning and model development
* A Flask RESTful API was built to programmaticly to identify the NER for each word in a sentence.

# Usage:
* Create a local conda virtual environment using the requirements.txt file [here](https://github.com/mrichtmyer/samsungNEXT_tech/blob/master/requirements.txt)
  - Install using: pip install -r requirements.txt and then use conda activate to activate the virtual environment
  
* [Data Cleaning Notebook](https://github.com/mrichtmyer/samsungNEXT_tech/blob/master/Code/NER_EDA.ipynb) showcases all data ingestion and cleaning pipelines used
  - Run all cells in this book locally
  
* [Modeling](https://github.com/mrichtmyer/samsungNEXT_tech/blob/master/Code/NER_ML.ipynb) documents a Naive Bayes model to classify
  - Run all cells in this book locally. The word_vectors matrix was not uploaded to Github as it is >100MB. This is needed for the RESTful API
  
* [API](https://github.com/mrichtmyer/samsungNEXT_tech/blob/master/Code/app.py) 


## API Usage
* Change directories into the /Code folder
* Run python app.py. This will serve up the Flask API in a local host
* In a separate terminal tab or window, SSH into this local host and post a query using this structure: 
  - curl -H "Content-Type: application/json" -X POST -d '"Jack lives in London"' http://127.0.0.1:5000/
  
# Next Steps:
1. Currently, this model is treating this task as a multi-class classification problem, and the multinomial naive bayes classification seems to work well enough. If I had more time, I would want to repeat this model in Tensorflow or PyTorch, which would likely be more robust, but also easier to continue scaling (e.g. use previous model weights and retrain on new data when the accuracy dips)
2. This model will need to be retrained if we add new annotated data as the DictVectorizer will contain new feature vectors (e.g. one hot encoding) for each new word. 
3. The current model is not necessarily sensitive to a particular language - each word is encoded, so the model is just trained on these encodings and the labels. In theory, adding different words (within or among languages) would just add more encodings. 

