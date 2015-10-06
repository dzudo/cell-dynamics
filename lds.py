import numpy as np
import numpy.random as npr
import random
import os
import matplotlib.pyplot as plt
import time
import math


plt.ion()

Y = []#np.matrix([100.])]

U = np.matrix([math.sin(i*math.pi/(20)) for i in range(101)]).T
#out +=  U
x_exp = [np.matrix([0.5, 0.1,0.3]).T]
y_exp = []


A = np.matrix([[1., 0.,0.], [0., 1.,0.],[0.,0.,1.]])
B = np.matrix([1., -1., 0.]).T
C = np.matrix([1.,-1.,0.])

for i in range(100):
	x_exp.append(np.matrix([[1., 1., 0.], [0.,1., 0.],
        [0.,0.,1.]]).dot(x_exp[i]) + (np.matrix([0.5, 1.,-0.003]).T).dot(U[i]))
	Y.append(np.matrix([1.,0.,0.]).dot(x_exp[i]) + random.gauss(0.,4.))

#out +=  x_exp
#out +=  Y	
	
Q = np.matrix([[0.1, 0.,0.], [0., 0.1, 0.],[0.,0.,0.0]])
R = np.matrix([4.0])

# STATE ESTIMATION

v1 = np.matrix([[1., 0.,0.], [0., 1.,0.],[0.,0.,1.]])

pi = np.matrix([0., 0.,0.]).T

t = [i for i in range(len(Y))]

y_res = []
for i in range(len(Y)):
	y_res.append(C.dot(x_exp[i]))

line1,line2 = plt.plot(t, [y[0,0] for y in Y], 'r--', t, [y[0,0] for y in y_res], 'b--')
plt.draw()


for n in range(500):
	
	
	out = ""
	out +=  "############### STAGE " + str(n) + " ################\n"

	#EXPECTATION PHASE
	J = []
	Vt1 = []
	#out +=  np.array(pi.T)
	X = [np.matrix(npr.multivariate_normal(np.array(pi.T).flat,v1)).T]
	V = [v1]
	K = []
	#KALMAN FILTERING
	for i in range(len(Y)-1):
		X.append(A.dot(X[-1]) + B.dot(U[i]))
		V.append(A.dot(V[-1]).dot(A.T) + Q)
		
		K.append(V[-1].dot(C.T).dot((C.dot(V[-1]).dot(C.T) + R).I))
		
		V[-1] -= K[-1].dot(C).dot(V[-1])
		X[-1] += K[-1].dot(Y[i+1] - C.dot(X[-1]))
	
	Vt1 = [(np.identity(V[-1].shape[0])-K[-1].dot(C)).dot(A).dot(V[-1])]	
	#KALMAN SMOOTHING
	for i in range(len(Y)-2, -1, -1):
		x_tplus1 = A.dot(X[i]) + B.dot(U[i])
		Vtplus1 = A.dot(V[i]).dot(A.T) + Q
		
		J.insert(0,V[i].dot(A.T).dot(Vtplus1.I))
		if len(J)>1:
			Vt1.insert(0,V[i].dot(J[0].T) + J[1].dot(Vt1[0] - A.dot(V[i])).dot(J[0].T))
		
		X[i] += J[0].dot(X[i+1]-x_tplus1)
		V[i] += J[0].dot(V[i+1]-Vtplus1).dot(J[0].T)
		
		
	
	#out +=  "Y:" + str([round(x[0,0],2) for x in Y])
	#out +=  "X1: " + str([round(x[0,0],2) for x in X])
	#out +=  "X2: " + str([round(x[1,0],2) for x in X])
	
	#MAXIMIZATION PHASE
	Cnew = (sum([Y[i].dot(X[i].T) for i in range(len(Y))])).dot((sum([V[i] + X[i].dot(X[i].T) for i in range(len(Y))])).I)
	
	Bnew = sum([X[i].dot(U[i-1].T) - A.dot(X[i-1]).dot(U[i-1].T) for i in range(1,len(Y))]).dot(sum([U[i-1].dot(U[i-1].T) for i in range(1,len(Y))]).I)
	
	Anew = sum([Vt1[i-1] + X[i].dot(X[i-1].T) - B.dot(U[i-1]).dot(X[i-1].T) for i in range(1, len(Y))]).dot(sum([V[i] + X[i].dot(X[i].T) for i in range(len(Y)-1)]).I)
	
	
	R=(sum([Y[i].dot(Y[i].T) - Cnew.dot(X[i]).dot(Y[i].T) for i in range(len(Y))]))/len(Y)
	
	#Q = (sum([V[i] + X[i].dot(X[i].T) for i in range(len(V))] ) -
    #    Anew.dot(sum([Vt1[i] + X[i+1].dot(X[i].T) - Bnew.dot(U[i]).dot(X[i].T) for i in range(len(Y)-1)])) )/(len(Y)-1)
	
	pi = X[0]
	v1 = V[0]
	
	A = Anew
	B = Bnew
	C = Cnew
	
	out += "\n"
	
	out +=  "A \n" + str(A) +  "\n"
	out +=  "B \n" + str(B) + "\n"
	out +=  "C \n" + str(C) + "\n"
	out +=  "Q \n" + str(Q) + "\n"
	out +=  "R \n" + str(R) + "\n"
	out +=  "pi \n" + str(pi) + "\n"
	out +=  "v1 \n" + str(v1) + "\n"
	
	out +=  "\n############### STAGE " +str(n) + " END #############\n"
	os.system('clear')
	print out
	x_exp = [X[0]]
	y_res = []
	for i in range(len(Y)):	
		y_res.append(C.dot(x_exp[-1]))
		x_exp.append(A.dot(x_exp[i]) + B.dot(U[i]))
		
	#print [y[0,0] for y in Y]
	#print [y[0,0] for y in y_res
	line1.set_ydata([y[0,0] for y in Y])
	line2.set_ydata([y[0,0] for y in y_res])
	plt.draw()
	#plt.savefig("res/iteration_" + str(n) + ".png")
	print "\n"
