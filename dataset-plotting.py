import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data=pd.read_csv('./heart.csv')

#----------Rename All Columns----------
data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})

#----------Age Analysis----------

#----------Show different Ages Graphs----------
sns.barplot(x=data.Age.value_counts()[:20].index,y=data.Age.value_counts()[:20].values)
plt.xlabel('Age')
plt.ylabel('Age Counter')
plt.title('Age Graph')
plt.show()

#----------Young,Middle,ElderlyAge Graph----------
young_ages=data[(data.Age>=29)&(data.Age<40)]
middle_ages=data[(data.Age>=40)&(data.Age<55)]
elderly_ages=data[(data.Age>55)]
sns.barplot(x=['young_ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counters')
plt.title('Different Ages State in Dataset')
plt.show()


#----------Gender Graphs----------
plt.title('Gender Graph')
sns.countplot(data.Sex)
plt.show()

#----------Chest Pain Type Analysis----------
data.Cp.value_counts()
sns.countplot(data.Cp)
plt.xlabel('Chest Type')
plt.ylabel('Count')
plt.title('Chest Type vs Count State')
plt.show()

#----------Thalach Analysis----------
data.Thalach.value_counts()[:20]
sns.barplot(x=data.Thalach.value_counts()[:20].index,y=data.Thalach.value_counts()[:20].values)
plt.xlabel('Thalach')
plt.ylabel('Count')
plt.title('Thalach Counts')
plt.xticks(rotation=45)
plt.show()

#----------Thal Analysis----------
data.Thal.value_counts()
sns.countplot(data.Thal)
plt.show()

#----------Result Analysis----------
#----------determine the age ranges of patients with and without sickness----------
age_unique=sorted(data.Age.unique())
age_counter_target_1=[]
age_counter_target_0=[]
for age in data.Age.unique():
    age_counter_target_1.append(len(data[(data['Age']==age)&(data.Target==1)]))
    age_counter_target_0.append(len(data[(data['Age']==age)&(data.Target==0)]))
plt.scatter(x=data.Age.unique(),y=age_counter_target_1,color='blue',label='Target 1')
plt.scatter(x=data.Age.unique(),y=age_counter_target_0,color='red',label='Target 0')
----------graph plotting----------
plt.legend(loc='upper right',frameon=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Result 0 & Result 1 State')
plt.show()
