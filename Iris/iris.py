import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/iris/Iris.csv")

data.head()

data.info()

data.drop('Id', axis=1, inplace=True)

plt.figure(figsize=(10,5))
plt.scatter(data[data.Species=='Iris-setosa']['SepalLengthCm'], data[data.Species=='Iris-setosa']['SepalWidthCm'], label='Iris-setosa')
plt.scatter(data[data.Species=='Iris-versicolor']['SepalLengthCm'], data[data.Species=='Iris-versicolor']['SepalWidthCm'], label='Iris-versicolor')
plt.scatter(data[data.Species=='Iris-virginica']['SepalLengthCm'], data[data.Species=='Iris-virginica']['SepalWidthCm'], label='Iris-virginica')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Sepal Length VS Width")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.scatter(data[data.Species=='Iris-setosa']['PetalLengthCm'], data[data.Species=='Iris-setosa']['PetalWidthCm'], label='Iris-setosa')
plt.scatter(data[data.Species=='Iris-versicolor']['PetalLengthCm'], data[data.Species=='Iris-versicolor']['PetalWidthCm'], label='Iris-versicolor')
plt.scatter(data[data.Species=='Iris-virginica']['PetalLengthCm'], data[data.Species=='Iris-virginica']['PetalWidthCm'], label='Iris-virginica')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length VS Width")
plt.legend()
plt.show()

data.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

data.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=data)

data.shape

data.head()
x_train, x_test, y_train, y_test = train_test_split(data.drop('Species', axis=1), data.Species, test_size=.33, random_state=101)

model_svm = svm.SVC()
model_svm.fit(x_train, y_train)
prediction = model_svm.predict(x_test)
acc_svm = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the SVM:', acc_svm)

model_logistic_reg = LogisticRegression()
model_logistic_reg.fit(x_train, y_train)
prediction = model_logistic_reg.predict(x_test)
acc_logistic_reg = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the Logistic Regression:', acc_logistic_reg)

model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(x_train, y_train)
prediction = model_decision_tree.predict(x_test)
acc_decision_tree = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the Decision Tree:', acc_decision_tree)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)
prediction = model_knn.predict(x_test)
acc_knn = metrics.accuracy_score(prediction, y_test)
print('The accuracy of the KNN:', acc_knn)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 'Decision Tree', 'KNN'],
    'Score': [acc_svm, acc_logistic_reg, acc_decision_tree, acc_knn]
    })

models

accuracies = []
x = [i for i in range(1, 11)]
for i in x:
    model = KNeighborsClassifier(n_neighbors=i) 
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracies.append(metrics.accuracy_score(prediction, y_test))
    
plt.figure(figsize=(10, 5))
plt.plot(x, accuracies)
plt.xticks(x)
plt.show()

plt.figure(figsize=(7,4))
sns.heatmap(data.corr(), annot=True)
plt.show()

sx_train, sx_test, sy_train, sy_test = train_test_split(data[["SepalLengthCm", "SepalWidthCm"]], data.Species, 
                                                        test_size=.33, random_state=101)
px_train, px_test, py_train, py_test = train_test_split(data[["PetalLengthCm", "PetalWidthCm"]], data.Species, 
                                                        test_size=.33, random_state=101)

model_svm = svm.SVC()
model_svm.fit(sx_train, sy_train)
prediction = model_svm.predict(sx_test)
acc_s_svm = metrics.accuracy_score(prediction, sy_test)
print('The accuracy of the SVM using Sepal:', acc_s_svm)

model_svm = svm.SVC()
model_svm.fit(px_train, py_train)
prediction = model_svm.predict(px_test)
acc_p_svm = metrics.accuracy_score(prediction, py_test)
print('The accuracy of the SVM using Petal:', acc_p_svm)

model_logistic_reg = LogisticRegression()
model_logistic_reg.fit(sx_train, sy_train)
prediction = model_logistic_reg.predict(sx_test)
acc_s_logistic_reg = metrics.accuracy_score(prediction, sy_test)
print('The accuracy of the Logistic Regresion using Sepal:', acc_s_logistic_reg)

model_logistic_reg = LogisticRegression()
model_logistic_reg.fit(px_train, py_train)
prediction = model_logistic_reg.predict(px_test)
acc_p_logistic_reg = metrics.accuracy_score(prediction, py_test)
print('The accuracy of the Logistic Regression using Petal:', acc_p_logistic_reg)

model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(sx_train, sy_train)
prediction = model_decision_tree.predict(sx_test)
acc_s_decision_tree = metrics.accuracy_score(prediction, sy_test)
print('The accuracy of the Decision Tree using Sepal:', acc_s_decision_tree)

model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(px_train, py_train)
prediction = model_decision_tree.predict(px_test)
acc_p_decision_tree = metrics.accuracy_score(prediction, py_test)
print('The accuracy of the Decision Tree using Petal:', acc_p_decision_tree)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(sx_train, sy_train)
prediction = model_knn.predict(sx_test)
acc_s_knn = metrics.accuracy_score(prediction, sy_test)
print('The accuracy of the KNN using Sepal:', acc_s_knn)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(px_train, py_train)
prediction = model_knn.predict(px_test)
acc_p_knn = metrics.accuracy_score(prediction, py_test)
print('The accuracy of the KNN using Petal:', acc_p_knn)

sepal_accs = [acc_s_svm, acc_s_logistic_reg, acc_s_decision_tree, acc_s_knn]
petal_accs = [acc_p_svm, acc_p_logistic_reg, acc_p_decision_tree, acc_p_knn]
accs = [sepal_accs, petal_accs]

models = pd.DataFrame(accs, index=['Using Sepal', 'Using Petal'], columns=['SVM', 'Logistic Reg', 'Decision Tree', 'KNN'])
models
