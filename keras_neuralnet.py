import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
#-------------Dataset Reading-------------
dataset = pd.read_csv("./heart.csv")
#-------------train & test-------------
predictors = dataset.drop("target",axis=1)
target = dataset["target"]
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)
#-------------keras implementation-------------
modelKeras = Sequential()
modelKeras.add(Dense(15,activation='relu',input_dim=13))
modelKeras.add(Dense(1,activation='sigmoid'))
modelKeras.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
modelKeras.fit(X_train,Y_train,epochs=2000)
Y_pred_nn = modelKeras.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
score_keras = round(accuracy_score(Y_pred_nn,Y_test)*100,2)
print("The accuracy score achieved using Keras is: "+str(score_keras)+" %")