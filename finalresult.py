import matplotlib.pyplot as plt
import seaborn as sns
#all algorithm scores
score_nBayes= 86.89
score_svMachine= 85.25
score_dTree= 86.89
score_rForest= 91.8
score_keras= 89.5
#final-score plot
scores = [score_nBayes,score_svMachine,score_dTree,score_rForest,score_keras]
algorithms = ["Naive Bayes","Support Vector Machine","Decision Tree","Random Forest","Neural Network"]
for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")
sns.set(rc={'figure.figsize':(16,9)})
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)