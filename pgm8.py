import numpy as np
from sklearn import preprocessing,model_selection,neighbors
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from matplotlib import style

df=pd.read_csv("iris.csv")
df.replace('setosa',1, inplace=True)
df.replace('versicolor',2, inplace=True)
df.replace('virginica',3, inplace=True)
#Missing Data Handling
df.replace('?',-9999,inplace=True)
#Define Attributes and Classes
X=np.array(df.drop(['species'],1))
Y=np.array(df['species'])
X_train,X_test,Y_train,Y_test= model_selection.train_test_split(X,Y,test_size=0.2)
plt.plot(X_train,Y_train,'b.')
# Define the classifier using panda library
clf=neighbors.KNeighborsClassifier()
# Save the model with the fit method
clf.fit(X_train,Y_train)
# use the test set and calculate the accuracy of the model
accuracy=clf.score(X_test, Y_test)
print("Accurancy:")
print(accuracy)
print("-------------------------------------------------")
example_measures = np.array([[4.7,3.2,2,0.2],[5.1,2.4,4.3,1.3]])
example_measures = example_measures.reshape(2,-1)
prediction = clf.predict(example_measures)
print("Prediction")
print(prediction)