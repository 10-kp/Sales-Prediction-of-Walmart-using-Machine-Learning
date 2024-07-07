# Sales-Prediction-using-Machine-Learning


Data Science Project in python to predict the sales for each department using historical markdown data from the Walmart dataset containing data of 45 Walmart stores.
The purpose of this project is to develop a predictive model and find out the sales of each product at a given Walmart store.
This project features a exploratory analysis and my predictive model was primarily based on linear regression

Predict which departments are affected with the holiday markdown events and the extent of impact.

We would also like to create a linear model to find a specific value for Weekly Sales that we want to predict. This line of best fit is intended to approximate further data points based on the line that we find in our training data.
Perform dimensionality reduction to improve prediction error by shrinkage in order to reduce overfitting.

### Business Understanding

#### Walmart is an American retail corporation that operates a chain of hypermarkets, discount department stores, and grocery stores.

![image](https://github.com/10-kp/Sales-Prediction-using-Machine-Learning/assets/70857174/63802b7d-30ec-419c-a8e4-8b2229cf18d6)

### Data Understanding

The data contains these features:
- Store - the store number
- Date - the week of sales
- Weekly_Sales - sales for the given store
- Holiday_Flag - whether the week is a special holiday week 1 – Holiday week 0 – Non-holiday week
- Temperature - Temperature on the day of sale
- Fuel_Price - Cost of fuel in the region
- CPI – Prevailing consumer price index
- Unemployment - Prevailing unemployment rate

## Import Libraries

Import the following into the code: 
- pandas as pd
- import numpy as np
- import seaborn as sns
- import matplotlib.pyplot as plt
- from matplotlib import dates
- from datetime import datetime
- from sklearn import datasets
- from sklearn.ensemble import RandomForestRegressor
- from sklearn.linear_model import LinearRegression
- from sklearn.model_selection import train_test_split
- from sklearn import metrics
