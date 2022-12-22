#!/usr/bin/env python
# coding: utf-8

# In[163]:


#the model
#f(x)=(w*x)+b
import copy
import numpy as np
w_initial = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_initial =785.1811367994083
def prediction (w,x,b):
    prediction = np.dot(w,x)+b
    return prediction


# In[164]:


#test
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = prediction(w_initial,x_vec, b_initial)
print( f_wb)


# In[109]:


X_train[2]


# In[165]:


#cost function
#J = 1/2m for i in range (number of rows) error**2

def calc_cost (x,y,w,b):
    cost = 0.0
    m= x.shape[0]
    for i in range(m):
        f_wb_i = np.dot(w,x[i]) + b
        cost = cost + (f_wb_i - y[i]) **2
    cost = cost / (2 * m)
    return cost


# In[166]:


cost = calc_cost(X_train, y_train, w_initial, b_initial)
print(f'Cost at optimal w : {cost}')


# In[167]:


#gradient decent
# (1 / m) * (model prediction - y ) * (x)
def calc_gradient (x,y,w,b):
    
    m,n = x.shape
    dj_dw = np.zeros((n,)) 
    dj_db = 0.0
    for i in range(m):
        error = (np.dot(w,x[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] +  error * x[i,j]
        dj_db = dj_db + error
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db,dj_dw 


# In[168]:


tmp_dj_db, tmp_dj_dw = calc_gradient(X_train, y_train, w_initial, b_initial)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


# In[171]:


#gradient decent cumulative function
def gradient_decent(x,y,w,b,calc_gradient,alpha,number_of_iterations):
    m , n = x.shape
    w_in = copy.deepcopy(w)  #avoid modifying global w within function
    b_in = b
    #dj_db , dj_dw = calc_gradient(x,y,w,b)
    for j in range(number_of_iterations):
        dj_db , dj_dw = calc_gradient(x,y,w_in,b)
        w_in = w_in - (alpha * dj_dw)
        b_in = b_in - (dj_db * alpha)
        #for i in range(m):
         #   w_in[i] = w_in[i] - (alpha * dj_dw[i])
        #b_in = b_in - (dj_db * alpha) 
    return w_in , b_in


# In[160]:


initial_w = np.zeros_like(w_initial)
initial_w


# In[172]:



# initialize parameters
initial_w = np.zeros_like(w_initial)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7

w_final, b_final = gradient_decent(X_train, y_train, initial_w, initial_b, calc_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")


# In[173]:


m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")


# In[ ]:




