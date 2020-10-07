
# ## Importing the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = pd.read_csv(R"D:\Data Reimagined\Task 1\INTERNET BANKING - Sheet1.csv")
data.head()

sns.countplot(x='Internet Banking Usage',data=data)


sns.countplot(x='Internet Banking Usage',hue='Gender',data=data)


sns.countplot(y="Area of living (District)", data=data)


sns.countplot(x='Internet Banking Usage',hue='Age',data=data)


sns.countplot(y="Profession", data=data)

sns.countplot(x='Lack of Knowledge',data=data)

plt.show()
