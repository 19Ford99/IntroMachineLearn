#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


DF = pd.read_csv("D3.csv") # Reads csv file and sets it to the variable D3

X1 = np.array(DF.values[:,0]) # 1st input column
X2 = np.array(DF.values[:,1]) # 2nd input column
X3 = np.array(DF.values[:,2]) # 3rd input column

y  = np.array(DF.values[:,3]) # Last column, output
m = len(y) # Number of values in dataset


# In[3]:


def compute_loss(X,y,theta):
#    **** Computes the loss function for linear regression ****
    h = X.dot(theta) # h = predictions
    errors = np.subtract(h,y)
    sqrErrors = np.square(errors)
    J = 1/(2*m) * np.sum(sqrErrors)
    return J

def gradient_descent(X,y,theta,alpha,iterations):
    loss_history = np.zeros(iterations)
    for i in range(iterations):
        h = X.dot(theta)
        errors = np.subtract(h,y)
        sum_delta = (alpha/m)*X.transpose().dot(errors);
        theta = theta - sum_delta;
        loss_history[i] = compute_loss(X,y,theta)
    return theta, loss_history
    
def predict(X, theta):
    return theta[0] + theta[1]*X[0] + theta[2]*X[1] + theta[3]*X[2]


# In[4]:


# Problem 1
# Inputs Vs Outputs
plt.rcParams['figure.figsize'] = [10, 4]
plt.figure()
plt.scatter(X1, y, color ='red', marker= '+', label = 'Training Data')
plt.xlabel("X1")
plt.ylabel("y")
plt.title("Figure 1: y Vs X1 ")
plt.grid()
plt.legend()

plt.figure()
plt.scatter(X2, y, color ='green', marker= '+', label = 'Training Data')
plt.xlabel("X2")
plt.ylabel("y")
plt.title("Figure 2: y Vs X2")
plt.grid()
plt.legend()

plt.figure()
plt.scatter(X3, y, color ='orange', marker= '+', label = 'Training Data')
plt.xlabel("X3")
plt.ylabel("y")
plt.title("Figure 3: y Vs X3")
plt.grid()
plt.legend();


# In[5]:


# Create a matrix with a single column of ones
X0 = np.ones((m,1))

# Using reshape function to convert X1,X2,X3 into a 2D Array
X1 = X1.reshape(m, 1)
X2 = X2.reshape(m, 1)
X3 = X3.reshape(m, 1)

# Using hstack() function to stack X_0,X_1,X_3 horizontally
X1 = np.hstack((X0, X1)) 
X2 = np.hstack((X0, X2)) 
X3 = np.hstack((X0, X3))


# In[6]:


theta = np.zeros(2)

# Compute the cost for theta values 
loss1 = compute_loss(X1, y, theta)
print('The cost for X1 =', loss1)
loss2 = compute_loss(X2, y, theta)
print('The cost for X2 =', loss2)
loss3 = compute_loss(X3, y, theta) 
print('The cost for X3 =', loss3)


# In[7]:


theta = [0. , 0.]
iterations = m; 
alpha = 0.1;
theta1, loss1_history = gradient_descent(X1, y, theta, alpha, iterations) 
print('Final value of theta1 =', theta1) 

iterations = m; 
alpha = 0.1;
theta2, loss2_history = gradient_descent(X2, y, theta, alpha, iterations) 
print('Final value of theta2 =', theta2)

iterations = m; 
alpha = 0.1;
theta3, loss3_history = gradient_descent(X3, y, theta, alpha, iterations) 
print('Final value of theta3 =', theta3)


# In[8]:


# Plots the model for X1
plt.figure()
plt.scatter(X1[:,1], y, color='red', marker= '+', label= 'Training Data') 
plt.plot(X1[:,1],X1.dot(theta1), color='blue', label='Linear Regression') 
plt.grid() 
plt.xlabel('X1') 
plt.ylabel('h (x)') 
plt.title('Linear Regression Fit for X1') 
plt.legend()

# Plots the model for X2
plt.figure()
plt.scatter(X2[:,1], y, color='green', marker= '+', label= 'Training Data') 
plt.plot(X2[:,1],X2.dot(theta2), color='blue', label='Linear Regression') 
plt.grid() 
plt.xlabel('X2') 
plt.ylabel('h (x)') 
plt.title('Linear Regression Fit for X2') 
plt.legend()

# Plots the model for X3
plt.figure()
plt.scatter(X3[:,1], y, color='orange', marker= '+', label= 'Training Data') 
plt.plot(X3[:,1],X3.dot(theta3), color='blue', label='Linear Regression') 
plt.grid() 
plt.xlabel('X3') 
plt.ylabel('h (x)') 
plt.title('Linear Regression Fit for X3') 
plt.legend();


# In[9]:


# Plots the loss history for X1
plt.figure()
plt.plot(loss1_history[0:len(loss1_history)], color='blue') 
plt.grid() 
plt.xlabel('Number of iterations') 
plt.ylabel('Cost (J)') 
plt.title('Convergence of gradient descent for X1')

# Plots the loss history for X2
plt.figure()
plt.plot(loss2_history[0:len(loss2_history)], color='blue') 
plt.grid() 
plt.xlabel('Number of iterations') 
plt.ylabel('Cost (J)') 
plt.title('Convergence of gradient descent for X2')

# Plots the loss history for X3
plt.figure()
plt.plot(loss3_history[0:len(loss3_history)], color='blue') 
plt.grid() 
plt.xlabel('Number of iterations') 
plt.ylabel('Cost (J)') 
plt.title('Convergence of gradient descent X3');


# In[10]:


# Problem 2
D3 = pd.read_csv("D3.csv") # Reads csv file and sets it to the variable D3

X1 = np.array(D3.values[:,0]) # 1st input column
X2 = np.array(D3.values[:,1]) # 2nd input column
X3 = np.array(D3.values[:,2]) # 3rd input column

y  = np.array(D3.values[:,3]) # Last column, output
m = len(y) # Number of values in dataset


# In[11]:


# Create a matrix with a single column of ones
X0 = np.ones((m,1))

# Using reshape function to convert X1,X2,X3 into a 2D Array
X1 = X1.reshape(m, 1)
X2 = X2.reshape(m, 1)
X3 = X3.reshape(m, 1)

# Using hstack() function to stack X_0,X_1,X_3 horizontally
X = np.hstack((X0, X1, X2, X3))


# In[12]:


theta = np.zeros(4)

# Compute the cost for theta values 
loss = compute_loss(X, y, theta)
print('The cost for X =', loss)


# In[13]:



iterations = m; 
alpha = 0.1;
theta, loss_history = gradient_descent(X, y, theta, alpha, iterations) 
print('Final value of theta =', theta)


# In[14]:


# Plots the loss history for X
plt.figure()
plt.plot(range(1, iterations + 1),loss_history, color='blue') 
plt.grid() 
plt.xlabel('Number of iterations') 
plt.ylabel('Cost (J)') 
plt.title('Convergence of gradient descent for X');


# In[15]:


# Prediction for new values

# new_X = (1,1,1)
new_X = np.array([1,1,1])
y_pred = predict(new_X, theta)
print("For values (1,1,1): ", y_pred)

# new_X = (2,0,4)
new_X = np.array([2,0,4])
y_pred = predict(new_X, theta)

print("For values (2,0,4): ", y_pred)

# new_X = (3,2,1)
new_X = np.array([3,2,1])
y_pred = predict(new_X, theta)

print("For values (3,2,1): ",y_pred)

