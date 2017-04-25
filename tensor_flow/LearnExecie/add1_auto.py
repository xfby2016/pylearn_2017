#-*- coding: utf-8 -*-
'''
Created on 24.4, 2017


Input:      varible

Output:     tensor

@author: xl
'''
import tensorflow as tf
pass
base = tf.Variable(0,name='counter')
constand1 = tf.constant(1)
add_bc = tf.add(base,constand1)
update = tf.assign(base,add_bc)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(base))
    #sess.run(update)
    #print(sess.run(update))
    for _ in range(3):
        sess.run(update)
        print(sess.run(base))