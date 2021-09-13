# Project Description: The best place for a new oil well
<p align="center">
<img src="https://user-images.githubusercontent.com/56832126/123948675-f945df00-d9a1-11eb-8914-83fb0301e7c3.png" width="500px">
</p>

In this project, as a Data Scientist for the OilyGiant mining company, our task is to find the best place for a new well.

## Steps to choose the location:
- Collect the oil well parameters in the selected region: oil quality and volume of reserves;
- Build a model for predicting the volume of reserves in the new wells;
- Pick the oil wells with the highest estimated values;
- Pick the region with the highest total profit for the selected oil wells.
- Data on wells from three regions have been provided. Parameters of each oil well in the region are already known. We will build a model that will help to pick the region with the highest profit margin. In the end, we will analyze potential profit and risks using the Bootstrapping technique.____


## Table of Contents:
- **Load and prepare the data**
- **Train and test the model for each region:**
  - Split the data into a training set and validation set at a ratio of 75:25.
  - Train the model and make predictions for the validation set.
  - Save the predictions and correct answers for the validation set.
  - Print the average volume of predicted reserves and model RMSE.
  - Analyze the results.
- **Prepare for profit calculation:**
  - Store all key values for calculations in separate variables.
  - Calculate the volume of reserves sufficient for developing a new well without losses. Compare the obtained value with the average volume of reserves in each region.
  - Provide the findings about the preparation for profit calculation step.
- **Write a function to calculate profit from a set of selected oil wells and model predictions:**
  - Pick the wells with the highest values of predictions.
  - Summarize the target volume of reserves in accordance with these predictions
  - Provide findings: suggest a region for oil wells' development and justify the choice. Calculate the profit for the obtained volume of reserves.
- **Calculate risks and profit for each region:**
  - Use the bootstrapping technique with 1000 samples to find the distribution of profit.
  - Find average profit, 95% confidence interval and risk of losses. Loss is negative profit, calculate it as a probability and then express as a percentage.
  - Provide findings: suggest a region for development of oil wells and justify the choice.


## Data Description

Geological exploration data for the three regions are stored in files:
  - geo_data_0.csv.
  - geo_data_1.csv. 
  - geo_data_2.csv.
  
  - id — unique oil well identifier
  - f0, f1, f2 — three features of points (their specific meaning is unimportant, but the features themselves are significant)
  product — volume of reserves in the oil well (thousand barrels).

## Conditions:

Only linear regression is suitable for model training (the rest are not sufficiently predictable).
When exploring the region, a study of 500 points is carried with picking the best 200 points for the profit calculation.
The budget for development of 200 oil wells is 100 USD million.

One barrel of raw materials brings 4.5 USD of revenue The revenue from one unit of product is 4,500 dollars (volume of reserves is in thousand barrels).
After the risk evaluation, keep only the regions with the risk of losses lower than 2.5%. From the ones that fit the criteria, the region with the highest average profit should be selected.

*The data is synthetic: contract details and well characteristics are not disclosed.*
