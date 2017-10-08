import numpy as np

def func(p,theta,m,tedat):	
	for i in range (0,10000):
		theta -= (alpha/len(x)) * np.dot(np.transpose(x) ,(np.dot(x,theta) -y));
		theta[1:len(theta)] -= np.dot(np.power(np.absolute(theta[1:len(theta)]),p-1),np.transpose(np.sign(theta[1:len(theta)]))) * p * lamda * (alpha/len(x))

	tedat = (tedat - mean2)/std2
	tmp = np.ones(len(tedat))
	tedat[:,0] = tmp

	if( m != 4):
		output = open('output_p'+str(m)+'.csv' , 'w')
	else:
		output = open('output.csv' , 'w')
	output.write("ID,MEDV\n")
	for i in range(0,len(sample)):
		output.write("%d,%s\n" %(i, np.dot(tedat[i],theta)))


data= np.genfromtxt("data/train.csv" , delimiter =',' , skip_header=True)
sample = np.genfromtxt("data/test.csv" , delimiter =',' , skip_header=True)

dat = data[:,0:-1]
dat1 = data[:,1:-1]
a1 = np.square(dat1)
b1 = np.multiply(a1,dat1) 

trdat1 = np.concatenate((dat,a1),axis=1)
trdat = np.concatenate((trdat1,b1),axis=1)

sample1 = sample[:,1:]
a2 = np.square(sample1)
b2 = np.multiply(a2,sample1)

tedat1 = np.concatenate((sample,a2),axis=1)
tedat = np.concatenate((tedat1,b2),axis=1)

mean1 = np.mean(trdat,axis=0)
std1 = np.std(trdat,axis =0)
mean2 = np.mean(tedat,axis=0)
std2 = np.std(tedat,axis =0)

x = (trdat - mean1)/std1
t= np.ones(len(x))
x[:,0]=t
y= data[:,-1]

theta = [1.0]*trdat.shape[1];
theta = np.transpose(theta);
thetaCon=theta;

alpha = 0.1
lamda=0.01
p=2
tedatCon=tedat;

func(1.25,theta,1,tedat);
theta=thetaCon;
tedat=tedatCon;

func(1.5,theta,2,tedat);
theta=thetaCon;
tedat=tedatCon;

func(1.75,theta,3,tedat);
theta=thetaCon;
tedat=tedatCon;

func(2,theta,4,tedat);
theta=thetaCon;
tedat=tedatCon;

# CLOSED-FORM Solution
temp=np.ones(len(dat));
dat[:,0]=temp;
theta_Closed= np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(dat),dat)),np.transpose(dat)),y); #closed-form solution
y_closed = np.dot(sample,theta_Closed)
# print(y_closed)
