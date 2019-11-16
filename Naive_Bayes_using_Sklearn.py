import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'UserFeature_02.csv', skipinitialspace=True, header=None)

features=df.values[:,:5]

target=df.values[:,6]

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.33, random_state = 10)

clf=GaussianNB()

clf.fit(features_train, target_train)

target_pred = clf.predict(features_test)

print("The Prediction is : ",target_pred)

print("The Accuracy is : ",accuracy_score(target_pred,target_test))
if(accuracy_score(target_pred,target_test)>50):
  print("Exceptable")
else:
  print("Not Exceptable")
