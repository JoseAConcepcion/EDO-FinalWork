import numpy as np
import scipy as sp
import sympy as sy

class prey_depredator_hollingTypeII:

    # Parametros
    r = 0
    beta = 0
    alpha = 0
    alpha1 = 0
    eta = 0
    eta1 = 0
    k = 0
    rho = 0
    m = 0
    mu = 0

    def __init__(self, params):
        # [r, alpha, beta, rho, m, mu, eta, alpha1, eta1, k]
        # lo ideal era hacer el vector con los parametros pero es ilegible en las formulas
        self.r = params[0]
        self.beta = params[1]
        self.alpha = params[2]
        self.alpha1 = params[3]
        self.eta = params[4]
        self.eta1 = params[5]
        self.k = params[6]
        self.rho = params[7]
        self.m = params[8]
        self.mu = params[9]
        #make params as symbols
        
        
    # EDO
    def f1(self, t: float, x: float, y: float, z: float) -> float:
        return self.r*x * (1 - x/self.k) - self.beta*x - self.alpha*x*z

    def f2(self, t: float, x: float, y: float, z: float) -> float:
        return self.beta*x - self.eta*y*z/(self.m + y) - self.mu*y

    def f3(self, t: float, x: float, y: float, z: float) -> float:
        return self.alpha1*x*z - self.rho*z*z - self.eta1*y*z/(self.m + y)

    def jacobi_matrix() -> np.ndarray:
        
        params = [sy.Symbol('r'), sy.Symbol('beta'), sy.Symbol('alpha'), 
                  sy.Symbol('alpha1'), sy.Symbol('eta'), sy.Symbol('eta1'), 
                  sy.Symbol('k'), sy.Symbol('rho'), sy.Symbol('m'), sy.Symbol('mu')]
        
        system = prey_depredator_hollingTypeII(params)

        t, x, y, z = sy.symbols('t x y z')

        matrix = sy.Matrix([system.f1(t, x, y, z), system.f2(t, x, y, z), system.f3(t, x, y, z)])
        matrix = sy.simplify(matrix.jacobian([x, y, z]))
        eigvalues = sy.simplify(matrix)
        a = matrix.eigenvals()
        return a ,matrix

    def __str__(self) -> str:
        string = "Parametros\n"
        string += "r = " + str(self.r) + "\n"
        string += "alpha = " + str(self.alpha) + "\n"
        string += "beta = " + str(self.beta) + "\n"
        string += "rho = " + str(self.rho) + "\n"
        string += "m = " + str(self.m) + "\n"
        string += "mu = " + str(self.mu) + "\n"
        string += "eta = " + str(self.eta) + "\n"
        string += "alpha1 = " + str(self.alpha1) + "\n"
        string += "eta1 = " + str(self.eta1) + "\n"
        string += "k = " + str(self.k) + "\n"

        string += "\nEDO\n"
        string += "dx/dt = " + "{0}*x*(1-x/{1}) - {2}*x - {3}*x*z".format(self.r,self.k,self.beta,self.alpha) + "\n"
        string += "dy/dt = " + "{0}*x - {1}*y*z/({2} + y) - {3}*y".format(self.beta,self.rho,self.rho,self.mu) + "\n"
        string += "dz/dt = " + "{0}*x*z - {1}*z^2 - {2}*y*z/(y + {3})".format(self.alpha1,self.rho,self.eta1, self.m) + "\n"

        return string
