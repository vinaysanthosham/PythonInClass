import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
session = tf.Session()

X = tf.constant([1,2,3,4],shape=[2,2]) #matrix X
Y = tf.constant([4,3,2,2],shape=[2,2]) #matrix Y
Z = tf.constant([3,6,9,1],shape=[2,2]) #matrix Z


A = tf.pow(X,2) #matrix of X
B = tf.add(A,Y) #Addition of matrix A and Y
C = tf.multiply(B,Z) #Mutliplication of B & Z

print(session.run(A)) #printing A
print(session.run(B)) #printing B
print(session.run(C)) #printing C

session.close() #closing session