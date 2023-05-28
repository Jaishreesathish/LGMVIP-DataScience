# TASK 1 - Iris Flowers Classification ML Project

This particular ML project is usually referred to as the “Hello World” of Machine Learning. The iris flowers dataset contains numeric attributes, and it is perfect for beginners to learn about supervised ML algorithms, mainly how to load and handle data. Also, since this is a small dataset, it can easily fit in memory without requiring special transformations or scaling capabilities.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import numpy as np
warnings.filterwarnings("ignore")

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
df = pd.read_csv('iris.data', names=columns)
df.head()

df.shape

df.info()

df.describe()

print(df.groupby('Class_labels').size())

df['Class_labels'].unique()

correlation=data.corr()
correlation
sns.heatmap(correlation,annot=True)
plt.show()

visual = df['Class_labels'].value_counts().plot.bar(title='Flower class distribution', color=['green','yellow','blue'])
visual.set_xlabel('class',size=20)
visual.set_ylabel('count',size=20)
plt.show()

df.hist()
plt.show()

sns.pairplot(df, hue="Class_labels")
plt.show()

from sklearn.model_selection import train_test_split

X = df.drop(['Class_labels'], axis=1)
Y = df['Class_labels']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("Y_train.shape:", X_train.shape)
print("Y_test.shape:", Y_test.shape)

from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()

model1.fit(X_train,Y_train)

y_pred=model1.predict(X_test)

y_pred

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(Y_test,y_pred)

logpred=model1.predict(X_train)
print("Training Accuracy using Logistic Regression=",accuracy_score(Y_train,logpred)*100)

logpred1=model1.predict(X_test)
print("Test Accuracy using Logistic Regression =",accuracy_score(Y_test,logpred1)*100)

import numpy as np
x_new=np.array([[2.5,4,1.3,6],[5.3,2.5,4.6,1.9],[4.9,2.2,3.8,1.1]])
prediction=model1.predict(x_new)
print("Prediction of:{}".format(prediction))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model2=SVC()
model2.fit(X_train,Y_train)

svmpred=model2.predict(X_train)
print("Training Accuracy using SVM =",accuracy_score(Y_train,svmpred)*100)

svmpred1=model2.predict(X_test)
print("Test Accuracy using SVM =",accuracy_score(Y_test,svmpred1)*100)

x_new=np.array([[2.5,4,1.3,6],[5.3,2.5,4.6,1.9],[4.9,2.2,3.8,1.1]])
prediction=model2.predict(x_new)
print("Prediction of:{}".format(prediction))

from sklearn.neighbors import KNeighborsClassifier
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(X_train,Y_train)


knnpred = model3.predict(X_train)
print("Training Accuracy using KNN =",accuracy_score(Y_train,knnpred)*100)

#print("Accuracy Score:",accuracy_score(Y_test,y_pred2))

knnpred1=model3.predict(X_test)
print("Test Accuracy using KNN =",accuracy_score(Y_test,knnpred1)*100)

x_new=np.array([[2.5,4,1.3,6],[5.3,2.5,4.6,1.9],[4.9,2.2,3.8,1.1]])
prediction=model3.predict(x_new)
print("Prediction of:{}".format(prediction))

from sklearn.naive_bayes import GaussianNB
model4 = GaussianNB()
model4.fit(X_train,Y_train)

gaussianpred = model4.predict(X_train)
print("Training Accuracy using Gaussian NB =",accuracy_score(Y_train,gaussianpred)*100)

gaussianpred1=model4.predict(X_test)
print("Test Accuracy using GaussianNB =",accuracy_score(Y_test,gaussianpred1)*100)

x_new=np.array([[2.5,4,1.3,6],[5.3,2.5,4.6,1.9],[4.9,2.2,3.8,1.1]])
prediction=model4.predict(x_new)
print("Prediction of:{}".format(prediction))

# I used four algorithms for training and testing the models.From the above test cases, It is clear that Logistic Regression and SVM techniques provide accurate results.
