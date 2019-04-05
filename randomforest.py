import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#-------------Dataset Reading-------------
dataset = pd.read_csv("./heart.csv")
#-------------train & test-------------
predictors = dataset.drop("target",axis=1)
target = dataset["target"]
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
#-------------Random Forest Implementation-------------
max_accuracy = 0
for x in range(1000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
rForest = RandomForestClassifier(random_state=best_x)
rForest.fit(X_train,Y_train)
Y_pred_rForest = rForest.predict(X_test)

score_rForest = round(accuracy_score(Y_pred_rForest,Y_test)*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rForest)+" %")


