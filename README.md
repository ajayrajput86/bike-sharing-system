# Project Name
> projectLinear-Regression-Bike-Sharing-Assignment

.


## Table of Contents
1. Importing all required lib
2. Reading raw data file and data cleaning
  - check for null value
  - check unique value in each column 
  - checking data type
  - Renaming few columns for better understanding and recaling feature name 
  - Dropping redundent columns
  - check data size 

3. Checking any outlier , as all max and min value of continues variable lie between (mean +- 3*std)  , eleminate posibility of outlier  

4. EDA : Exploratory data Analysis  and Inference 
5. Encoding/mapping the catagorical column
6.Dummy data creation for catagorical variable
7. Data prepdation for model input 
  - Split dat into train and test
  - Data scaling with min max on continues variable

8. Feature engineering
    - Recursive feature elimination  for feature selection
    - Ordinary least squares model building 
    - Calculate VIF 
    - Remove feature based on P value and VIF for model tuning 
9.Check for co-efficent agaist each feature
10.Residual Analysis
	  - Checking Homoscedasticity with train data result
	  - R2 and Adjested_R2
11. Predictions of test variable Using the developed Model
	  - scaling Min Max test data
	  - Preparing input for model 
	  - predicting target variable 
	  - Checking Homoscedasticity with test result
  - y_test and y_pred spread
12. Result accuracy prediction
  - Calculation of R2_test and Adj_R2_test for test result



<!-- You can include any other section that is pertinent to your problem -->

## General Information

A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 


In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

Which variables are significant in predicting the demand for shared bikes.
How well those variables describe the bike demands
Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 


Business Goal:
You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 


day.csv have the following fields:
	
	- instant: record index
	- dteday : date
	- season : season (1:spring, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2018, 1:2019)
	- mnth : month ( 1 to 12)
	- holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : temperature in Celsius
	- atemp: feeling temperature in Celsius
	- hum: humidity
	- windspeed: wind speed
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered



<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
Demand is high for next year 2019
Season 3 has highest demand.
Month 5 to 10 has peak demand.
Year start and end has low demand and mid month has high demand
The clear weathershit highest booking
High booking on non holiday

Based on the final model,  top 3 features contributing significantly towards
explaining the demand of the shared bikes
1.temp
2.Year
3.Light_snowrain



## Technologies Used
# importing required libreries 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

