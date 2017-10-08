import pandas as pd
import numpy as np
import csv,ast

test_data = pd.read_csv('kaggle_test_data.csv')
test_id=test_data['id']
test_data.drop('id',axis =1,inplace = True)

test_data.replace(['?'],['None'], inplace=True)
test_data.fillna(test_data.mode().iloc[0],inplace=True)

cat_columns_test = test_data.select_dtypes(['object']).columns
for col in cat_columns_test:
    test_data[col]=test_data[col].astype('category')
    test_data[col]=test_data[col].cat.codes

f =open('weights.txt','r')
itr = iter(f)
lst = ast.literal_eval(next(itr))
lst = list(map(lambda x:list(map(float,x)),lst))
b1 = np.array(lst)
lst = ast.literal_eval(next(itr))
lst = list(map(lambda x:list(map(float,x)),lst))
b2 = np.array(lst)
lst = ast.literal_eval(next(itr))
lst = list(map(lambda x:list(map(float,x)),lst))
W1= np.array(lst)
lst = ast.literal_eval(next(itr))
lst = list(map(lambda x:list(map(float,x)),lst))
W2= np.array(lst)

test_data = test_data.as_matrix()
test_data=(test_data-np.mean(test_data,axis=0))/np.std(test_data,axis=0)

    # Forward propagation
z1 = test_data.dot(W1) + b1
a1 = np.tanh(z1)
z2 = a1.dot(W2) + b2
z2 = np.clip( z2, -500, 500 )
exp_scores = np.exp(z2)
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
test_salary = np.argmax(probs, axis=1)

test_id=test_id.as_matrix()
output = open('predictions.csv','w')
output.write("id,salary\n")
for i in range(0,len(test_salary)):
	output.write("%d,%s\n" %(test_id[i], test_salary[i]))
