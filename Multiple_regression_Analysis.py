
# # INTERNET BANKING USAGE

# ## Importing the relevant libraries


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()


# ## Loading the raw data

data = pd.read_csv(R"D:\Data Reimagined\Task 1\INTERNET BANKING - Sheet1.csv")
data.head()

data.columns

data.describe(include='all')

x= data.corr().values
x

sns.heatmap(data.corr())
plt.show()

data = pd.get_dummies(data, columns =['Gender', 'Age', 'Profession', 'Area of living (District)',
       'Monthly Income ', 'Reliability ',
       'Speed and availability on all devices', 'Attitude ', 'Ease of use',
       'Time consuming', 'Lack of Knowledge'],drop_first = True)
data.head()



data.columns

x = data[['Gender_Male', 'Age_31 - 40', 'Age_41 - 50', 'Age_Above 50',
       'Profession_Entrepreneur/Businessman', 'Profession_Student',
       'Area of living (District)_Anuradhapura',
       'Area of living (District)_Badulla',
       'Area of living (District)_Colombo', 'Area of living (District)_Galle',
       'Area of living (District)_Gampaha',
       'Area of living (District)_Hambantota',
       'Area of living (District)_Jaffna',
       'Area of living (District)_Kalutara', 'Area of living (District)_Kandy',
       'Area of living (District)_Kegalle',
       'Area of living (District)_Kurunegala',
       'Area of living (District)_Matale',
       'Area of living (District)_Monaragala',
       'Area of living (District)_Mullativu',
       'Area of living (District)_Nuwara Eliya',
       'Area of living (District)_Puttalam',
       'Area of living (District)_Ratnapura',
       'Area of living (District)_Vavuniya',
       'Monthly Income _50,000 - 100,000', 'Monthly Income _Above 100,000',
       'Monthly Income _Below 25,000', 'Reliability _Highly Satisfied',
       'Reliability _Highly dissatisfied', 'Reliability _Neutral',
       'Reliability _Satisfied',
       'Speed and availability on all devices_Highly Satisfied',
       'Speed and availability on all devices_Highly dissatisfied',
       'Speed and availability on all devices_Neutral',
       'Speed and availability on all devices_Satisfied',
       'Attitude _Highly Satisfied', 'Attitude _Highly dissatisfied',
       'Attitude _Neutral', 'Attitude _Satisfied',
       'Ease of use_Highly Satisfied', 'Ease of use_Highly dissatisfied',
       'Ease of use_Neutral', 'Ease of use_Satisfied', 'Time consuming_Yes',
       'Lack of Knowledge_Yes']]
y = data['Internet Banking Usage']


x_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

reg = LinearRegression()
reg.fit(x_train,y_train)


y_hat = reg.predict(x_train)


sns.distplot(y_train - y_hat)

# Include a title
plt.title("Residuals PDF", size=18)


reg.score(x_train,y_train)


reg.intercept_

reg.coef_

# Create a regression summary where we can compare them with one-another
reg_summary = pd.DataFrame(x.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary



