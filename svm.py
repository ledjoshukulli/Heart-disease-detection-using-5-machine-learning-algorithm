import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#-------------Dataset Reading-------------
dataset = pd.read_csv("./heart.csv")
#-------------train & test-------------
predictors = dataset.drop("target",axis=1)
target = dataset["target"]
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=1)
#-------------Support Vector Machine Implementation-------------
svMachine = svm.SVC(kernel='linear')
svMachine.fit(X_train, Y_train)
Y_pred_svMachine = svMachine.predict(X_test)
score_svMachine = round(accuracy_score(Y_pred_svMachine,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svMachine)+" %")
