#!/usr/bin/env python
# coding: utf-8


# ## Open the data file and have a look at the general information. 

# In[67]:


# import all required dependencies
import pandas as pd

# import NLTK and stemmer for working with texts
import nltk
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
from collections import Counter

#read the data and assign it to the credit_data variable
credit_data = pd.read_csv('credit_scoring_eng.csv')


# In[68]:


#get familiarize with datas by sample 5 rows of data
credit_data.sample(5)


# In[69]:


#rename the column name
credit_data.rename(columns={'dob_years':'customer_age'}, inplace=True)


# In[70]:


# get know about the shape of data and info about each column's datatype
print('Table has {} rows, {} columns  and {} observations in total.'
      .format(credit_data.shape[0], credit_data.shape[1], (credit_data.shape[0]*credit_data.shape[1])))
print()
print()
print('Datatypes of columns are as follow:')
print('----------')
credit_data.info()


# ## Data preprocessing

# ### Processing missing values

# In[71]:


# check total number of missing observations in the data
credit_data.isnull().sum().sum()


# In[72]:


# check percentage of missing values in each column
credit_data.isnull().sum()*100/len(credit_data)


# In[73]:


#retrieve data with rows that contain missing value on any of the columns
# and randomly select 10 or more rows with the sample() method
credit_data[credit_data.isnull().any(axis=1)].sample(10)


# In[74]:


# check 'income_type' of rows with missing values 
credit_data[credit_data.isnull().any(axis=1)]['income_type'].value_counts()


# In[75]:


# group observations according to income_type and gender, 
#hen retrieve mean and median of total_income
credit_data.groupby(['income_type','gender']).agg(
    {'total_income': ['mean', 'median']})


# In[76]:


# find how many different values in gender column
credit_data.gender.value_counts()


# In[77]:


# change value of XNA to M
credit_data.loc[credit_data.gender == 'XNA', 'gender'] = 'M'


# In[78]:


# check if the issue was fixed
credit_data.gender.value_counts()


# In[79]:


# after fixing XNA, group observations according to income_type and gender, 
#hen retrieve mean and median of total_income
credit_data.groupby(['income_type','gender']).agg(
    {'total_income': ['mean', 'median']})


# In[80]:


# fill missing values in total_income according to the condition of income_type
# and gender columns
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='F')&
                 (credit_data.income_type == 'retiree')),'total_income']=218529.2465
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='M')&
                 (credit_data.income_type == 'retiree')),'total_income']=20918.3620
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='F')&
                 (credit_data.income_type == 'employee')),'total_income']=20898.4980
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='M')&
                 (credit_data.income_type == 'employee')),'total_income']=25945.7880
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='F')&
                 (credit_data.income_type == 'civil servant')),'total_income']=21917.1980
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='M')&
                 (credit_data.income_type == 'civil servant')),'total_income']=29754.3915
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='F')&
                 (credit_data.income_type == 'business')),'total_income']=25731.3245
credit_data.loc[((credit_data.total_income.isnull())&(credit_data.gender=='M')&
                 (credit_data.income_type == 'business')),'total_income']=31498.3750
credit_data.loc[((credit_data.total_income.isnull())&
                 (credit_data.income_type == 'entrepreneur')),'total_income']=79866.1030


# In[81]:


#check if all missings were fixed in total_income column
credit_data.total_income.isnull().sum()


# In[82]:


credit_data[credit_data.days_employed > 0]['income_type'].value_counts()


# In[83]:


credit_data.loc[credit_data.days_employed <= 0]['income_type'].value_counts()


# In[84]:


# get mean and median values of days_employed according to income_type
credit_data.groupby(['income_type']).agg({'days_employed': ['mean', 'median']})


# In[85]:


#fill missing values in days_employed columns with the median values
credit_data.loc[((credit_data.days_employed.isnull())&
                 (credit_data.income_type == 'retiree')),'days_employed']=365213.306266
credit_data.loc[((credit_data.days_employed.isnull())&
                 (credit_data.income_type == 'employee')),'days_employed']=-1574.202821
credit_data.loc[((credit_data.days_employed.isnull())&
                 (credit_data.income_type == 'civil servant')),'days_employed']=-2689.368353
credit_data.loc[((credit_data.days_employed.isnull())&
                 (credit_data.income_type == 'business')),'days_employed']=-1547.382223
credit_data.loc[((credit_data.days_employed.isnull())&
                 (credit_data.income_type == 'entrepreneur')),'days_employed']=366413.652744


# In[86]:


# check if all missing values of data was fixed
credit_data.isnull().sum().sum()


# ### Data type replacement

# In[87]:


credit_data.info()


# In[88]:


# get insight about the descriptive statistics of columns which stores numerical values!
credit_data.describe()


# In[89]:


#loop thourgh give columns and change their data type into int8
for col in ['children', 'customer_age', 'education_id', 'family_status_id', 'debt']:
    credit_data[col] = credit_data[col].astype('int8')

# convert into int64 from float64 for two columns
credit_data.days_employed = credit_data.days_employed.astype('int64')
credit_data.total_income = credit_data.total_income.astype('int64')


# In[90]:


# check number of different value in children column
credit_data.children.value_counts()


# In[91]:


#change values of the children column with values of -1 and 20 to 1 and 2 respectively.
credit_data.loc[(credit_data['children'] == -1), 'children'] = 1
credit_data.loc[(credit_data['children'] == 20), 'children'] = 2

#check if it worked
credit_data.children.value_counts()


# In[92]:


#check if all worked
credit_data.info()


# ### Processing duplicates

# In[93]:


#find number of duplicated rows in data
credit_data.duplicated().sum()


# In[94]:


# print unique values in categorical columns, to check if there is low/uppercase issue
print(credit_data.education.unique())
print()
print(credit_data.family_status.unique())
print()
print(credit_data.income_type.unique())
print()
print(credit_data.purpose.unique())


# In[95]:


# convert all values in education column into lower case.
credit_data.education = credit_data.education.str.lower()
credit_data.education.unique()


# In[96]:


# collect purposes in all rows, and then make a big list containing all words
# go through all words and get their stems. Sort stems according to their frequency with Counter
words_list =[]
for sentence in credit_data.purpose:
    words = nltk.word_tokenize(sentence)
    for word in words:
        stemmed = stemmer.stem(word)
        words_list.append(stemmed)
Counter(words_list).most_common()


# In[97]:


# construct a function, that goes thorough pupose column and 
#cagetorize them according to the stems
def purpose_shorter(purpose):
    """takes purpose, makes lsit of words in it and then obtains stems of those words, 
    while producing list of stems. Later it checks each purpose if it contains 
    specific stem, then assing categories: house, car, education and wedding"""
    stems =[]
    words = nltk.word_tokenize(purpose)
    for word in words:
        stemmed = stemmer.stem(word)
        stems.append(stemmed)
    if ('hous' in stems) or ('properti' in stems) or ('estat' in stems):
        return 'house'
    if 'car' in stems:
        return 'car'
    if ('educ'in stems) or ('univers' in stems):
        return 'education'
    if 'wed' in stems:
        return 'wedding'
    else:
        return float('NaN')

        
# check if the function is working properly
print(purpose_shorter(credit_data.purpose[0]))


# In[98]:


#apply the function the data and make a new column
credit_data['purpose_category'] = credit_data.purpose.apply(purpose_shorter)

#check new data
credit_data.sample(5)


# In[99]:


#drop purpose column
credit_data.drop('purpose', inplace=True, axis=1)


# In[100]:


# check new state of data
credit_data.head(3)


# In[101]:


credit_data.duplicated().sum()


# In[102]:


# what percentage of rows duplicated
credit_data.duplicated().sum()*100/len(credit_data)


# In[103]:


credit_data.drop_duplicates(inplace=True)


# In[104]:


# get an insight about the diestrubition of values in total_income column
credit_data.total_income.describe()


# In[105]:


#write a function that makes category according to the amount of total income
def income_categorizer(income):
    if income < 17089:
        return 'low'
    if income < 23458.5:
        return 'middle'
    if income < 32050.75:
        return 'high'
    else:
        return 'very high'

#apply the fucntion to the data
credit_data['income_level'] = credit_data['total_income'].apply(income_categorizer)


# In[106]:


# check new data with income category
credit_data.head()


# In[107]:


# constuct a pivot table with counting customer number 
#who defauled or not according to their number of children
pivot_table_kid = credit_data.pivot_table(
    index='debt', columns='children', aggfunc={'debt':['count']})
pivot_table_kid


# In[108]:


#calculate the percentage of customers who defaulted out of total customers
# and add this percantge as a third row of the pivot table
pivot_table_kid.loc['defaulted %'] = pivot_table_kid.loc[1]*100/(pivot_table_kid.loc[0] + pivot_table_kid.loc[1])
pivot_table_kid


# In[109]:


# make a pivot table as in previous task
pivot_marital = credit_data.pivot_table(index='debt', columns='family_status',  aggfunc={'debt':['count']})
pivot_marital.loc['defaulted %'] = pivot_marital.loc[1]*100/(pivot_marital.loc[0] + pivot_marital.loc[1])
pivot_marital


# In[110]:


# make a pivot table as in previous task
pivot_income = credit_data.pivot_table(index='debt', columns='income_level',  aggfunc={'debt':['count']})
pivot_income.loc['defaulted %'] = pivot_income.loc[1]*100/(pivot_income.loc[0] + pivot_income.loc[1])
pivot_income


# In[111]:


# make a pivot table as in prevous task
pivot_purpose = credit_data.pivot_table(index='debt', columns='purpose_category',  aggfunc={'debt':['count']})
pivot_purpose.loc['defaulted %'] = pivot_purpose.loc[1]*100/(pivot_purpose.loc[0] + pivot_purpose.loc[1])
pivot_purpose
