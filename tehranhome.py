
#***Import library**

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import SCORERS, mean_absolute_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import math

#"""***Data Information***"""
# read CSV file
df=pd.read_csv('housePriceTehran.csv',header=0)

#show head of the dataset
df.head()

#show number of column and rows of the dataset
df.shape

# drow a plot to compare the number of house that have elevator, warehouse or parking
fig, ax = plt.subplots(ncols=3, figsize=(18,6))

colors = [['#ADEFD1FF', '#00203FFF'], ['#97BC62FF', '#2C5F2D'], ['#F5C7B8FF', '#FFA177FF']]
explode = [0, 0.2]
columns = ['Parking', 'Warehouse', 'Elevator']
for i in range(3):
        data = df[columns[i]].value_counts()
        ax[i].pie(data, labels=data.values, explode=explode, colors=colors[i], shadow=True)
        ax[i].legend(labels=data.index, fontsize='large')
        ax[i].set_title('{} distribution'.format(columns[i]))

Parking = df['Parking'].value_counts()

Warehouse = df['Warehouse'].value_counts()

Elevator = df['Elevator'].value_counts()
fig1, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,15)) 

labels = 'True', 'False'
ax1.pie([Parking[1], Parking[0]], labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.set_title("Parking")

labels = 'True', 'False'
ax2.pie([Warehouse[1], Warehouse[0]], labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax2.set_title("Warehouse")

labels = 'True', 'False'
ax3.pie([Elevator[1], Elevator[0]], labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax3.set_title("Elevator")

plt.show()

df.describe()

df.info()

# show the price for each address
df['Address'].value_counts().nlargest(5)

df.groupby('Address').mean()['Price'].nlargest(5).reset_index()

addressLPIR = df.groupby('Address').mean()['Price'].nlargest(5).reset_index()
sns.barplot(x="Address",
           y="Price",
           data=addressLPIR)

# show the price in $ for each address
df.groupby('Address').mean()['Price(USD)'].nlargest(5).reset_index() #USD

addressLP = df.groupby('Address').mean()['Price(USD)'].nlargest(5).reset_index()
sns.barplot(x="Address",
           y="Price(USD)",
           data=addressLP)

# show a plot to count hourse with the number of their room
sns.countplot(x="Room",data=df)

#show a diagram how price change when the romm number change
sns.lineplot(data = df, x = 'Room', y ='Price' )

# show a piot to compare number of the noom with price and elevator
sns.jointplot(data=df,x='Room', y='Price', hue='Elevator')

# show a piot to compare number of the noom with price and parking
sns.jointplot(data=df,x='Room', y='Price', hue='Parking')

# show a plot to compare number of the rooms with price
plt.scatter(df['Room'], df['Price(USD)'],  color='blue')
plt.xlabel("Number of Rooms")
plt.ylabel("Price(USD)")
plt.ticklabel_format(useOffset=False, style='plain')
plt.show()

df[(df['Room']== 0) & (df['Price(USD)']> 2000000)]

df[df['Address']== 'Tajrish']



##"""**Data Preprocessing**"""

df.dtypes.to_frame()

df.info()

df.drop("Price", inplace=True, axis = 1)
df.head()

# show the number of null value for each feature
df = df.fillna("Unknown")
missing_data = df.isnull().sum()
print(missing_data)

df.Parking = df.Parking.astype(int)
df.Warehouse = df.Warehouse.astype(int)
df.Elevator = df.Elevator.astype(int)
df.Area = df.Area.str.replace(',' , '').astype(int)

df.nlargest(5,'Area')

# delete record that have Area more than 2000000
df.drop( df[df['Area'] >= 2000000].index , inplace=True)
df.info()

# show null value
df.isnull().values.any()

df.isnull().sum()

# change feature Address to feature code Address
Address_df = df.groupby('Address').mean()['Price(USD)'].reset_index()

Address_df = Address_df.sort_values(by=['Price(USD)']).reset_index()

Address_df.insert(0, 'codedAddress', range(1, 194))
df1 = pd.Series(Address_df.codedAddress.values,index=Address_df.Address).to_dict()
df["Code_Adress"] = df["Address"].map(df1)

df.head()
# create a new feature 
df2=df["Price(USD)"]/df["Area"] 

MS = df.groupby('Area').mean()['Price(USD)'].reset_index()

MS = MS.sort_values(by=['Price(USD)']).reset_index()

df.insert(1, "Metr_Usd",df2)

df.head()

# show the plot to compare metr_usd and Area
plt.scatter(df['Metr_Usd'], df['Area'],  color='blue')
plt.xlabel("Metr_Usd")
plt.ylabel("Area")
plt.ticklabel_format(useOffset=False, style='plain')
plt.show()

df.nlargest(5,'Metr_Usd')

# show the metr_used for each address group by Address
addressLP = df.groupby('Address').mean()['Metr_Usd'].nlargest(5).reset_index()
sns.barplot(x="Address",
           y="Metr_Usd",
           data=addressLP)

# show the metr_used for each Metr-usd group by Address
addressLP = df.groupby('Address').mean()['Metr_Usd'].nsmallest(4).reset_index()
sns.barplot(x="Address",
           y="Metr_Usd",
           data=addressLP)

# show the metr_used for each Area group by Address
addressLP = df.groupby('Area').mean()['Metr_Usd'].nlargest(4).reset_index()
sns.barplot(x="Area",
           y="Metr_Usd",
           data=addressLP)

df[df['Area'] == 287]

# show scatter to compare metr_usd and Price
plt.scatter(df['Metr_Usd'], df['Price(USD)'],  color='blue')
plt.xlabel("Metr_Usd")
plt.ylabel("Price(USD)")
plt.ticklabel_format(useOffset=False, style='plain')
plt.show()

# show jointplt to compare Room and Metr_usd and parking
sns.jointplot(data=df,x='Room', y='Metr_Usd', hue='Parking')

# show jointplt to compare Room and Metr_usd and parking
sns.jointplot(data=df,x='Room', y='Metr_Usd', hue='Elevator')

df.drop("Price(USD)", axis = 1)

plt.figure(figsize=(8,5))
sns.displot(df['Metr_Usd'] , bins=30 , kde=True )

# show corellation between features
sns.heatmap(df.corr(), annot=True,cmap='Greens')

X = df.drop(['Price(USD)', 'Address'],axis=1) 
y = df['Price(USD)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_train.shape , X_test.shape

# create a Model
model= LinearRegression() 
model.fit(X_train , y_train )

pd.DataFrame(model.coef_, X.columns, columns=['Coeficient'])

# show the MAE MSE RMSE
y_pred=model.predict(X_test) 
MAE= metrics.mean_absolute_error(y_test, y_pred)
MSE= metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(MSE) 

pd.DataFrame([MAE, MSE, RMSE], index=['MAE', 'MSE', 'RMSE'], columns=['Metrics'])

test_residuals=y_test-y_pred

# show some plot to show predicted value to test datas
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Y-Test $')
plt.ylabel('Y-Pred$')

sns.scatterplot(x=y_test, y=test_residuals)
plt.axhline(y=0, color='c', ls='--')
plt.xlabel('Y-Test $')
plt.ylabel('residuals$')

sns.jointplot (x=y_test,y=test_residuals,data=df,kind='reg')

sns.displot(test_residuals, bins=45,kde=True)

r2_score(y_test,y_pred)

y_train_p=model.predict(X_train) 
r2_score(y_train,y_train_p)