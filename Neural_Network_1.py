################################################################################
#Date: October 21, 2017
#Programmer: K. Walter Schademan
#Note: code origin: https://iamtrask.github.io/2015/07/12/basic-python-network/
#Name: I'm Mister Meseeks Neural Network
#Description: proof of concept program for Neural Network design leading into 
#    more complicated neural networks
################################################################################

import numpy as np

#Sigmoid function returns a value between 0 and one for every x negative values 
#are return as less than .5 and positive values returned as above .5
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
#input data set
X = np.array([ [0,1,1],
               [0,0,1],
               [1,0,1],
               [1,1,1] ])
               
#outupt data set read from a file
Y = np.array([[0,1,1,0]]).T

#create and write to a file for the output array
def writeto(filename, array):
    file = open(filename, "w")
    for i in range(4):
        file.write(str(array[i][0]))
        file.write('\n')
    file.close()

#open the output array file and use the stored values to redefine the output 
#array
def readfrom(filename, array):
    with open(filename) as file:
        content = file.readlines()
    content = [float(x) for x in content]
    file.close()
    array = np.array([content]).T
    return array

Y = readfrom("outputfile", Y)

#seed random numbers to make calculation
np.random.seed(1)

#initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1,)) - 1

for iter in xrange(10000):
    
    #forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    
    #how much did we miss?
    l1_error = Y - l1
    
    #multiply how much we missed by the 
    #slope of the Sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
    
    #update weights
    syn0 += np.dot(l0.T,l1_delta)

writeto("outputfile", l1)  
print "Output after training:"
print l1