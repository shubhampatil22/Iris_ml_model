import pandas as pd

from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pickle

df=pd.read_csv(r"C:\Users\shubh\OneDrive\Desktop\Datasets\IRIS.csv")

print(df)
x=df.drop('species',axis=1)
y=df['species']

le=LabelEncoder()
y=le.fit_transform(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)
print(y_train)
classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print("Accuracy",metrics.accuracy_score(y_test,y_pred)*100)

pickle.dump(classifier,open('iris.pkl','wb'))

