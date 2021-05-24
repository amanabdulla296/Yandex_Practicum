#!/usr/bin/env python
# coding: utf-8

# # Analyzing borrowers’ risk of defaulting
# 
# Your project is to prepare a report for a bank’s loan division. You’ll need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# 
# Your report will be considered when building a **credit scoring** of a potential customer. A ** credit scoring ** is used to evaluate the ability of a potential borrower to repay their loan.
# 
# Here is brief steps we are going to implement:
# - import necessary libraries
# - open and get familier ourselves with the data
# - check if there are missing values
# - fix missin values
# - check and fix duplicates
# - categorize data according to the questions
# - answer the questions from the description of project

# ---

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


# > it is better experience to have sample rows from data rather than just head of data, therefore ```sample()``` method is applied.
# 
# > ```dob_years``` column name is not clear at a first glance, but in the description of data it was provided that it is an age of customers. If so, let's change the column name to ```customer_age```. Other column names seems clear!

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


# ### Conclusion
# 
# The required Python library - ```Pandas``` is imported and the provided data is opened successfully. With the printing of the head of the table, it has been observed that all columns stated in the job description are present! But the name of the ```dob_years``` column changed to ```customer_age``` for the sake of understanding.
# 
# The data contains  ```12``` columns, where 5 of them contains ```int64``` and another 5 contains ```object``` as well as 2 contains ```float64``` data types. There are ```21525``` rows in data, which make ```238300``` observations in total.

# ---

# ## Data preprocessing

# ### Processing missing values

# In[71]:


# check total number of missing observations in the data
credit_data.isnull().sum().sum()


# > It is ovserved that there are ```4348``` missing observations in whole table, which makes just 1,8% of total observations. Before doing anything with this missing data, let's check which columns have missing value and is there any pattern related to missing values!

# In[72]:


# check percentage of missing values in each column
credit_data.isnull().sum()*100/len(credit_data)


# > Only ```days_employed```and ```total_income``` columns have missing values. Number of missing values in both of them are exactly equal. Seems like they are related!

# In[73]:


#retrieve data with rows that contain missing value on any of the columns
# and randomly select 10 or more rows with the sample() method
credit_data[credit_data.isnull().any(axis=1)].sample(10)


# In[74]:


# check 'income_type' of rows with missing values 
credit_data[credit_data.isnull().any(axis=1)]['income_type'].value_counts()


# > It can be seen that when the value of ```days_employed``` is missing, the value of ```total_income``` is also missing at the same rows. At first glance, it gives an impression that if the person is not employed, then he/she will not have an income and therefore both of them are simultaneously empty. But, when we check the ```income_type``` column, it is stated that most of these people are employee or so, who should have an income. That means either they were left intentionally blank or they are due to a technical error.

# > Around 10% of rows in ```days_employed``` and ```total_income``` columns are missing. We can not throw them away. Let's try to fill them with the help of other columns.

# In[75]:


# group observations according to income_type and gender, 
#hen retrieve mean and median of total_income
credit_data.groupby(['income_type','gender']).agg(
    {'total_income': ['mean', 'median']})


# > It is obvious that incomes of people doing different jobs are not equal and another sad truth is that average income of males and females are also not equal.
# 
# > Another issue with the data is that ```gender``` column has a strange value of "XNA", let's fix it first.

# In[76]:


# find how many different values in gender column
credit_data.gender.value_counts()


# > only one row has value of 'XNA' for ```gender``` column. Most probably this is typos. The good thing is that its median value is very close to males' median. So we can assume it as a male and change its value to 'M'

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


# > when we compare mean and median values, it can be seen that mostly there a big difference. That is due to the skewness of total income. Therefore, let's use median values of ```total_income``` grouped by ```income_type``` and ```gender``` columns in oder to fill missing values. 

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


# > Missing values of ```total_income``` was fixed. Now we need to fill missing values of ```days_employed``` column. We can apply exactly the same method as used to fill ```total_income```. However ```days_employed``` columns values are weird, there are negative values and very large positive numbers like 395302. Let's try to understand what are they, why they look like that?

# In[82]:


credit_data[credit_data.days_employed > 0]['income_type'].value_counts()


# In[83]:


credit_data.loc[credit_data.days_employed <= 0]['income_type'].value_counts()


# > When we analyze the ```days_employed``` column together with ```income_type``` column together. We have observed that, ```days_emploayed``` contains large positive numbers only in case when ```income_type``` is either retiree or unemployed! Otherwise, i.e. for any other ```income_type``` they are negative numbers. From here, it can be concluded that "-" sign was used to differentiate between employed vs unemployed.
# 
# > Moreover, when we devide positive numbers to 365 in order to get years, we got unrealistic years as employed.
# 
# **At this point it is not clear what the numbers in ```days_employed``` actually mean.**
# 
# > However, these column is not important currently. Let's simply fill the missing values as we did ```total_income``` column (but omitting ```gender``` column).

# In[84]:


# get mean and median values of days_employed according to income_type
credit_data.groupby(['income_type']).agg({'days_employed': ['mean', 'median']})


# > Here also, mean and median values differentiate a lot due to outlier. Therefore let's use median values to fill missing values.

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


# ### Conclusion
# 
# There were missing values in ```days_employed``` and ```total_income``` columns. In total 10% of values were missing in each of the columns. Missing values were occurring in the same rows for the two columns. 
# The total income of different job types and gender was not equal. Moreover, there was an outlier in the data which resulted in different mean and median values. Hence, missing values of the ```total_income``` column are filled according to the median values of total income depending on ```income_type``` and ``` gender``` columns.
# Missing values of ```days_employed``` column was filled with median values days employed depending on ```income_type```. However, it was observed that **retiree** and **unemployed** customers assigned with relatively large positive numbers, in contrast, **employee**, **business** etc customers assigned with negative numbers. Actually, negative numbers can be converted to years by taking absolute value and dividing the value with 365. However, a positive number resulted in very large unrealistic years. So at this point, no further work done on the ```days_employed``` column.

# ---

# ### Data type replacement

# In[87]:


credit_data.info()


# In[88]:


# get insight about the descriptive statistics of columns which stores numerical values!
credit_data.describe()


# > From the descriptive statistics and information about columns, we can see that max/min numbers for ```children, customer_age, education_id, family_status_id, debt``` columns are between -128 and 127. That means instead of using ```int64``` data type, we can use ```int8``` which will save memory (currently 2.0+ MB).
# 
# > ```days_employed``` and ```total_income``` columns also have a floating point data type. For the sake of simplicity, lets get rid of the points or simply convert them into integer numbers.
# 
# > Also we have an issue with ```children``` columns, which says mininum value is -1 and max is 20. These values are unrealistic. Seems like a typos. 
# 
# >Let's fix these issues one by one. 

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


# > Now memory decreased from 2.0+MB to 1.3+MB and data type of columns changed. By the way, it is easy to read the ```days_employed``` and ```total_income``` columns.

# ### Conclusion
# 
# Data type of ```'children', 'customer_age', 'education_id', 'family_status_id', 'debt'``` columns changed from ```int64``` to ```int8``` to save memory. Also, it was hard to read values of ```days_employed and total_income``` columns due to the floating points. These columns' data type also changed to ```int64``` from ```float64```. There are artefacts observed in ```childrens``` column, such as -1 or 20 children. This was assumed as typing mistakes and these values changed to 1 and 2, respectively.

# ---

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


# > It is obvious that education column has a lowercase and uppercase words. We should convert them all into lowercase
# 
# > Also, in ```purpose``` columns, they purposes are described with different words. For that we will use NLTK library and make a new column which contains purpose categories.

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


# > Now we can drop intial ```purpose``` column, in that way we can better see the duplicates in our data. Because, maybe purpose of the applicant is recorded with different words, such as "house","real estate", "buying house" or "housing" etc.

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


# ### Conclusion
# 
# In the beginning, there were only **54** duplicates. But after converting all strings into lowercase and organizing purpose into categories, we have **405** duplicated rows.
# In our data, we do not have a unique identifier for each customer. That makes things a little bit confusing. But the good thing is we have 12 columns in total, which lowers the probability of different customers having the same indicators. Additionally, duplicated rows are less than 2% of total rows. Based on these we can drop all duplicated rows.

# ---

# ### Categorizing Data

# > According to the questions asked to us, we need to categorize our ```total_income``` column. Note that we have already categorized the ```purpose```` column in previous section.

# In[104]:


# get an insight about the diestrubition of values in total_income column
credit_data.total_income.describe()


# > Let's categorize ```total_income``` data using its quartiles. 1st quartile (25%) would be **low**, 2nd quartile (50%) - **medium**, 3rd quartile (75%) - **high** and anything else higher than 75% would be **very high** income categories.

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


# ### Conclusion
# 
# Categorization of the purposes was performed in the previous task. So here we have categorized total_income each customer get according to if they are below or above the specific quartile of the data. Briefly:
# 
# 1st quartile (25%) - **low**
# 
# 2nd quartile (50%) - **medium**
# 
# 3rd quartile (75%) - **high** 
# 
# higher than 75% - **very high**.

# ---

# ## Answering the questions

# - Is there a relation between having kids and repaying a loan on time?

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


# ### Conclusion
# 
# Most of the customers don't have any children and they have the lowest percentage of customers who defaulted. It is hard to compare data in this way because the population of customers with a high number of kids (e.g. there are only 329 people with three kid, while 12829 people without the kid) are significantly low than people who do not have a kid. Consequently, we can say that defaulting is not strongly related to having a kid.

# - Is there a relation between marital status and repaying a loan on time?

# In[109]:


# make a pivot table as in previous task
pivot_marital = credit_data.pivot_table(index='debt', columns='family_status',  aggfunc={'debt':['count']})
pivot_marital.loc['defaulted %'] = pivot_marital.loc[1]*100/(pivot_marital.loc[0] + pivot_marital.loc[1])
pivot_marital


# ### Conclusion
# 
# Based on the percentages of customers who defaulted, we can say that people who are **unmarried** or **civil partnership** has a higher defaulting rate. Maybe, that kind of people feels less responsibility :). In contrast, **widow** or **divorced** people are more likely to pay their debts on time.

# - Is there a relation between income level and repaying a loan on time?

# In[110]:


# make a pivot table as in previous task
pivot_income = credit_data.pivot_table(index='debt', columns='income_level',  aggfunc={'debt':['count']})
pivot_income.loc['defaulted %'] = pivot_income.loc[1]*100/(pivot_income.loc[0] + pivot_income.loc[1])
pivot_income


# ### Conclusion
# 
# There is no clear correlation between income level and defaulting. However, based on the percentages, we can say that people with **very high** and **low** income are more responsible for their debts. Whereas **middle** and **high** income people more likely to default.

# - How do different loan purposes affect on-time repayment of the loan?

# In[111]:


# make a pivot table as in prevous task
pivot_purpose = credit_data.pivot_table(index='debt', columns='purpose_category',  aggfunc={'debt':['count']})
pivot_purpose.loc['defaulted %'] = pivot_purpose.loc[1]*100/(pivot_purpose.loc[0] + pivot_purpose.loc[1])
pivot_purpose


# ### Conclusion
# 
# People who have borrowed money for construction or buying a house are less likely to default compared to other purposes. While people who had a purpose of **car** or **education** has a higher risk of default.

# ---

# ## General conclusion
# 
# The data about bank customer's debt repaying was successfully opened and analyzed. Missing values were detected, which might be intentionally left blank or technical error, and these missing values were filled using median values depending on values of other indicators. The number of duplicates increased when categorical columns are changed to lower case and the purpose column was changed to a shorter purpose category. Each question asked initially was answered depending on the data. However, there was not drastic relationship observed.

# ## Project Readiness Checklist
# 
# Put 'x' in the completed points. Then press Shift + Enter.

# - [x]  file open;
# - [x]  file examined;
# - [x]  missing values defined;
# - [x]  missing values are filled;
# - [x]  an explanation of which missing value types were detected;
# - [x]  explanation for the possible causes of missing values;
# - [x]  an explanation of how the blanks are filled;
# - [x]  replaced the real data type with an integer;
# - [x]  an explanation of which method is used to change the data type and why;
# - [x]  duplicates deleted;
# - [x]  an explanation of which method is used to find and remove duplicates;
# - [x]  description of the possible reasons for the appearance of duplicates in the data;
# - [x]  data is categorized;
# - [x]  an explanation of the principle of data categorization;
# - [x]  an answer to the question "Is there a relation between having kids and repaying a loan on time?";
# - [x]  an answer to the question " Is there a relation between marital status and repaying a loan on time?";
# - [x]   an answer to the question " Is there a relation between income level and repaying a loan on time?";
# - [x]  an answer to the question " How do different loan purposes affect on-time repayment of the loan?"
# - [x]  conclusions are present on each stage;
# - [x]  a general conclusion is made.
