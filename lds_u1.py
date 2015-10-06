import numpy as np
import numpy.random as npr
import random
import os
import matplotlib.pyplot as plt
import time
import math
import pickle


mng=plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.ion()

LENX = 6
LENU = 6
LENY = 2


Y = pickle.load(open("UY", "r"))#[]#np.matrix([100.])]
Yl = Y[:400]
Yt = Y[400:]

U = pickle.load(open("U2", "r"))#np.matrix([math.sin(i*math.pi/(20)) for i in range(101)]).T

Ul = U[:400]
Ut = U[400:]
#out +=  U


A = np.matrix(np.identity(LENX))#np.matrix([[1., 0.,0.], [0., 1.,0.],[0.,0.,1.]])
B = np.matrix(np.eye(LENX,LENU))#np.matrix([1., -1., 0.]).T
C = np.matrix(np.eye(LENY,LENX))#[1.,-1.,0.])

#out +=  x_exp
#out +=  Y	
	
Q = np.matrix(np.identity(LENX)*0.00001)#np.matrix([[0.1, 0.,0.], [0., 0.1, 0.],[0.,0.,0.0]])
R = np.matrix(np.identity(LENY)*0.1)#np.matrix([4.0])

# STATE ESTIMATION

v1 = np.matrix(np.identity(LENX))#np.matrix([[1., 0.,0.], [0., 1.,0.],[0.,0.,1.]])

pi = np.matrix(np.ones(LENX)*0.01).T

t = [i for i in range(len(Y))]

#y_res = []
#for i in range(len(Y)):
#	y_res.append(C.dot(x_exp[i]))

plt.subplot(311)

liney11,liney12 = plt.plot(t, [y[0,0] for y in Y], 'r--', t, [y[0,0] for y in Y], 'b--')
plt.subplot(312)
liney21,liney22 = plt.plot(t, [y[1,0] for y in Y], 'r--', t, [y[1,0] for y in Y], 'b--')
plt.subplot(313)
liney31,liney32 = plt.plot(t, [math.atan2(y[1,0],y[0,0]) for y in Y], 'r--', t,
        [math.atan2(y[1,0],y[0,0]) for y in Y], 'b--')


plt.draw()

d_avg = []
d_var = []
d_std = []

for n in range(300):
	
	
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
	for i in range(len(Yl)-1):
		X.append(A.dot(X[-1]) + B.dot(U[i]))
		V.append(A.dot(V[-1]).dot(A.T) + Q)
		
		K.append(V[-1].dot(C.T).dot((C.dot(V[-1]).dot(C.T) + R).I))
		
		V[-1] -= K[-1].dot(C).dot(V[-1])
		X[-1] += K[-1].dot(Y[i+1] - C.dot(X[-1]))
	
	Vt1 = [(np.identity(V[-1].shape[0])-K[-1].dot(C)).dot(A).dot(V[-1])]	
	#KALMAN SMOOTHING
	for i in range(len(Yl)-2, -1, -1):
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
	Cnew = (sum([Y[i].dot(X[i].T) for i in range(len(Yl))])).dot((sum([V[i] +
        X[i].dot(X[i].T) for i in range(len(Yl))])).I)
	
	Bnew = sum([X[i].dot(U[i-1].T) - A.dot(X[i-1]).dot(U[i-1].T) for i in
        range(1,len(Yl))]).dot(sum([U[i-1].dot(U[i-1].T) for i in range(1,len(Yl))]).I)
	
	Anew = sum([Vt1[i-1] + X[i].dot(X[i-1].T) - B.dot(U[i-1]).dot(X[i-1].T) for
        i in range(1, len(Yl))]).dot(sum([V[i] + X[i].dot(X[i].T) for i in
            range(len(Yl)-1)]).I)
	
	
	R=(sum([Y[i].dot(Y[i].T) - Cnew.dot(X[i]).dot(Y[i].T) for i in range(len(Yl))]))/len(Y)
	
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
	#out +=  "Q \n" + str(Q) + "\n"
	out +=  "R \n" + str(R) + "\n"
	out +=  "pi \n" + str(pi.T) + "\n"
	#out +=  "v1 \n" + str(v1) + "\n"
	
	out +=  "\n############### STAGE " +str(n) + " END #############\n"
	os.system('clear')
	print out
	x_exp = [X[0]]
	y_res = []
	for i in range(len(Y)):	
		y_res.append(C.dot(x_exp[-1]))
		x_exp.append(A.dot(x_exp[i]) + B.dot(U[i]))
		
    	diffs= [math.sqrt((Y[i][0,0] - y_res[i][0,0])**2 + (Y[i][1,0] - y_res[i][1,0])**2) for i in range(len(Y))]
	d_avg.append(np.average(diffs))
	d_std.append(np.std(diffs))
	d_var.append(np.var(diffs))

	#print [y[0,0] for y in Y]
	#print [y[0,0] for y in y_res
	liney11.set_ydata([y[0,0] for y in Y])
	liney12.set_ydata([y[0,0] for y in y_res])
    
	liney21.set_ydata([y[1,0] for y in Y])
	liney22.set_ydata([y[1,0] for y in y_res])

	liney31.set_ydata([math.atan2(y[1,0],y[0,0]) for y in Y])
	liney32.set_ydata([math.atan2(y[1,0],y[0,0]) for y in y_res])

	plt.draw()

	if n == 10:
		plt.savefig("100_" + str(n) + ".png")
	print "\n"

plt.savefig("res_data/result_u2_" + str(LENX) + ".png")

plt.clf()
t = np.arange(len(d_std))

plt.subplot(311)
plt.plot(t, d_avg, color='r')
plt.yscale('log')

plt.subplot(312)
plt.plot(t, d_std, color='g')
plt.yscale('log')

plt.subplot(313)
plt.plot(t, d_var, color='b')
plt.yscale('log')

plt.draw()

plt.savefig("res_data/result_u2_" + str(LENX) + "_stat.png")

model = {"A" : A, "B" : B, "C" : C, "R" : R, "Q" : Q, "pi" : pi, "v1" : v1}
pickle.dump(model, open("res_data/result_u2_" + str(LENX) + ".pik", 'w+'))

pickle.dump({'avg' : d_avg, 'var' : d_var, 'std' : d_std},
        open("res_data/result_u2_" + str(LENX) + "_stat.pik", 'w+'))




