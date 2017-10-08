import pandas as pd
import numpy as np

def build_model(X,Y,nn_input_dim,nn_output_dim,nn_hdim,epsilon,regLambda):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    model = {}
     
    for i in range(0,1000):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        z2 = np.clip( z2, -500, 500 )
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        delta1 = probs
        delta1[range(training_examples), Y] -= 1
        dW2 = (a1.T).dot(delta1)
        db2 = np.sum(delta1, axis=0, keepdims=True)
        delta2 = delta1.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        dW2 += regLambda * W2
        dW1 += regLambda * W1
 
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2
         
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return model

data = pd.read_csv('train.csv', delimiter=',')
Y= data['salary']

X = data
X.drop(['salary','id'],axis =1,inplace = True)

X.replace(['?'],['None'], inplace=True)
X.fillna(X.mode().iloc[0],inplace=True)

cat_columns = X.select_dtypes(['object']).columns
for col in cat_columns:
	X[col]=X[col].astype('category')
	X[col]=X[col].cat.codes

training_examples = len(X) 
nn_input_dim = 14 
nn_output_dim = 2 
 
epsilon = 0.01 
regLambda = 0.01 

X=X.as_matrix()
X=(X-np.mean(X,axis=0))/np.std(X,axis=0)

mean = np.mean(X,axis = 0)
std = np.std(X,axis=0)

nn_hdim = 50
model = build_model(X,Y,nn_input_dim,nn_output_dim,nn_hdim,epsilon,regLambda)

f =open('weights.txt','w')
f.write(str(model['b1'].tolist())+'\n')
f.write(str(model['b2'].tolist())+'\n')
f.write(str(model['W1'].tolist())+'\n')
f.write(str(model['W2'].tolist())+'\n')