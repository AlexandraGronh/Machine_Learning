
#=================
# Humidity forecast
# Normal and multivariate regression of weather data
# in ipython3
#=================

reset

# Hi Alex how are you? Everything good?

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from numpy import transpose

dataset = pd.read_csv('/home/alexandra/Desktop/Machine_Learning/datasets/weather_data/weatherHistory.csv')

dataset.describe()	# data overview

	dataset.isnull().any()	# check for NaNs
	# and fill potential NaNs (True above):
	dataset = dataset.fillna(method='ffill')

	# Fig. 1, exploration, regression pattern:
	dataset.plot(x='Humidity', y='Temperature (C)', style='o')	

	# Fig. 2, exploration, density distribution:
	plt.figure(figsize=(15,10))
	plt.tight_layout()
	seabornInstance.distplot(dataset['Humidity'])


	# defining dependent and independent variables
	# 'attributes' = the independent variables = x
	# 'labels' = dependent variables whose values are to be predicted = y


# definitions: predicting humidity by temperature
# 1) normal linear regression:
#X = dataset['Temperature (C)'].values.reshape(-1,1)
# 2) or multivariate regression:
X_all = dataset[['Temperature (C)','Wind Speed (km/h)', 'Humidity', 'Wind Speed (km/h)', 'Loud Cover', 'Pressure (millibars)']]
X = dataset[['Temperature (C)','Wind Speed (km/h)', 'Humidity', 'Wind Speed (km/h)', 'Loud Cover', 'Pressure (millibars)']].values
y = dataset['Humidity'].values.reshape(-1,1)


# splitting into 80% training and 20% test data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# training the algorithm:
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# checking results of linear regression:
# intercept:
print('intercept is: ')
print(regressor.intercept_)
# slope:
print('slope is: ')
print(regressor.coef_)


# checking for multivariate regression which parameter has main impact:
regressor.coef_2=transpose(regressor.coef_);
coeff_df = pd.DataFrame(regressor.coef_2, X_all.columns, columns=['Coefficient'])  
coeff_df



# make prediction on test data (to check algorithm accuracy of prediction of percentage score):
y_pred = regressor.predict(X_test)


# validation of prediction (comparing actual X_test with precidcted values):
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

# validation visually:
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#plotting data and regression line:
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# performance validation / might want to compare different algorithms here:
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))







