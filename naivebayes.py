import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#-------------Dataset Reading-------------
dataset = pd.read_csv("./heart.csv")
#-------------train & test-------------
from sklearn.model_selection import train_test_split
predictors = dataset.drop("target",axis=1)
target = dataset["target"]
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=2)
#-------------Naive Bayesian Implementation-------------
from sklearn.naive_bayes import GaussianNB
nBayes = GaussianNB()
nBayes.fit(X_train,Y_train)
Y_pred_nBayes = nBayes.predict(X_test)
score_nBayes = round(accuracy_score(Y_pred_nBayes,Y_test)*100,2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nBayes)+" %")