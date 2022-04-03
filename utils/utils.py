import numpy as np

class Utils(object):

    def __init__(self):
        pass
    
    def gradiant_descent(self ,gradiant , X ,y, alpha ,start ,iter, epcelon = 10e-5 ,):
        Xk = start
        k = 0
        while(np.all(np.diff(Xk , X))< epcelon or k > iter):
            dk  = -1 * gradiant(X , y , Xk)
            Xk =  Xk + alpha*dk
            k +=1
        return Xk