# Project description
The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. The goal is to train a model to automatically detect negative reviews. You'll be using a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews. **It will need to reach an F1 score of at least 0.85.**

![image](https://user-images.githubusercontent.com/56832126/130929741-8386f5ab-7a72-4401-bbcc-bf811dbdb69f.png)

## Project Instructions
- Load the data.
- Preprocess the data, if required.
- Conduct an EDA and make your conclusion on the class imbalance.
- Preprocess the data for modeling.
- Train at least three different models for the given train dataset.
- Test the models for the given test dataset.
- Compose a few of your own reviews and classify them with all the models.
- Check for differences between the testing results of models in the above two points. Try to explain them.
- Present your findings.


## Data Description
The data is stored in the imdb_reviews.tsv file. The data was provided by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

Here's the description of the selected fields:
``review`` : the review text
``pos`` : the target, '0' for negative and '1' for positive
``ds_part`` : 'train'/'test' for the train/test part of dataset, correspondingly

There are other fields in the dataset. Feel free to explore them if you'd like.
