#-*- coding: utf-8 -*-
'''
Created on 20.4, 2017


Input:      test

Output:     test

@author: xl
'''
import numpy as np

import tensorflow as tf
#__all__ = tf
import matplotlib.pyplot as plt
#creat data:  y=0.33x+0.25
data = []
num_data = 1000
for i in range(1000):
    x1 = np.random.normal(0,0.55)
    y1 = 0.33*x1+0.25+np.random.normal(0,0.05)
    data.append([x1,y1])
    pass
#print(data)
# get the sample data
x_data = [v[0] for v in data]
y_data = [v[1] for v in data]
print(x_data)
#plt.scatter(x_data,y_data,c='r',s = 20,alpha=0.7)
#plt.show()
w = tf.Variable(tf.random_uniform([1],-1.0,1.0),name='w')
b = tf.Variable(tf.zeros([1]),name='b')
pre_y = w*x_data+b
#print(w,b)
print(pre_y)
loss = tf.reduce_mean(tf.square(pre_y-y_data),name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss,name='train')
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#print init w&b
print('w = ',sess.run(w),'b = ',sess.run(b),'loss = ',sess.run(loss))
#train 20 times
for step in range(50):
    sess.run(train)
    print('w = ', sess.run(w), 'b = ', sess.run(b),'loss = ',sess.run(loss))
    pass
writer = tf.train.SummaryWriter('./tem',sess.graph)
pass
plt.scatter(x_data,y_data,c='r')
plt.plot(x_data,sess.run(w)*x_data+sess.run(b))
plt.ylabel('y_data')
plt.xlabel('x_data')
plt.show()



