import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


data = pd.read_csv('train.csv', delimiter=',')
Y= data['salary']
# print(y)

X = data
X.drop(['salary','id'],axis =1,inplace = True)
# print(X)

X.replace(['?'],['None'], inplace=True)
X.fillna(X.mode().iloc[0],inplace=True)

cat_columns = X.select_dtypes(['object']).columns
for col in cat_columns:
	X[col]=X[col].astype('category')
	X[col]=X[col].cat.codes
# print(X)

test_data = pd.read_csv('kaggle_test_data.csv')
test_id=test_data['id']
test_data.drop('id',axis =1,inplace = True)

test_data.replace(['?'],['None'], inplace=True)
test_data.fillna(test_data.mode().iloc[0],inplace=True)

cat_columns_test = test_data.select_dtypes(['object']).columns
for col in cat_columns_test:
    test_data[col]=test_data[col].astype('category')
    test_data[col]=test_data[col].cat.codes

X=X.as_matrix()
X=(X-np.mean(X,axis=0))/np.std(X,axis=0)

test_data = test_data.as_matrix()
test_data=(test_data-np.mean(test_data,axis=0))/np.std(test_data,axis=0)

classifiers = [
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
  	]

for i in range(0,len(classifiers)):
	clf = classifiers[i]
	clf.fit(X,Y)
	result = clf.predict(test_data)
	output = open('predictions_'+str(i+1)+'.csv' , 'w')
	output.write("id,salary\n")
	for j in range(0,len(result)):
		output.write("%d,%s\n" %(test_id[j], result[j]))



