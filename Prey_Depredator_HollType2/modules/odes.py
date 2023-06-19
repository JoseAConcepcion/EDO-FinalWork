import numpy as np
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

    tol = 1e-8  # tol para las comparaciones aritmeticas
    isSymbolic = False  # para los calculos simbolicos

    def __init__(self, params, isSimbolic=False):
        # [r, alpha, beta, rho, m, mu, eta, alpha1, eta1, k]
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
        self.isSymbolic = isSimbolic
        return

    # EDO
   
    def f1(self, t: float, x: float, y: float, z: float) -> float:
        return x*(self.r*(1 - x/self.k) - self.beta - self.alpha*z)
        

    def f2(self, t: float, x: float, y: float, z: float) -> float:
        return self.beta*x - y *(self.eta*z/(self.m + y) - self.mu)

    def f3(self, t: float, x: float, y: float, z: float) -> float:
        factor = -1 if (type(x)!=sy.Symbol and type(y)!=sy.Symbol) \
                        and (np.isclose(x,y) and np.isclose(x,0))  \
                    else 1
        
        return self.alpha1*x*z + factor*z*z*(self.rho - self.eta1/(self.m + y))

    def jacobi_matrix(self) -> sy.Matrix:
        t , x, y, z = sy.symbols('t x y z')
        jacobian = sy.Matrix([self.f1(t, x, y, z),self.f2(t, x, y, z),self.f3(t, x, y, z)]).jacobian([x, y, z])
        return jacobian

    def eigvalues(self): # probleams en los valores propios
        t ,x , y ,z = sy.symbols('t x y z')
        J = self.jacobi_matrix()
        sy.assumptions()
        return sy.Matrix.eigenvals(J)
        
    def jacobi_matrix_symbolic() -> sy.Matrix:
        params = [sy.Symbol('r'), sy.Symbol('beta'), sy.Symbol('alpha'),
                  sy.Symbol('alpha1'), sy.Symbol('eta'), sy.Symbol('eta1'),
                  sy.Symbol('k'), sy.Symbol('rho'), sy.Symbol('m'), sy.Symbol('mu')]
        system = prey_depredator_hollingTypeII(params, True)

        # t , x ,y ,z = sy.symbols('t x y z')
        # dx = sy.Eq(system.f1(t,x,y,z), 0)
        # dy = sy.Eq(system.f2(t,x,y,z), 0)
        # dz = sy.Eq(system.f3(t,x,y,z), 0)

        # solution = sy.solve([dx, dy, dz], [x, y, z])

        print(solution)
        
        return system.jacobi_matrix()

    def __str__(self) -> str:
        string = ""
        string += "\nEDO\n"
        string += "dx/dt = " + \
            "{0}*x*(1-x/{1}) - {2}*x - {3}*x*z".format(self.r,
                                                       self.k, self.beta, self.alpha) + "\n"
        string += "dy/dt = " + \
            "{0}*x - {1}*y*z/({2} + y) - {3}*y".format(self.beta,
                                                       self.rho, self.rho, self.mu) + "\n"
        string += "dz/dt = " + "{0}*x*z + {1}*z^2 - {2}*z^2/(y + {3})".format(
            self.alpha1, self.rho, self.eta1, self.m) + "\n"
        return string

    def params(self) -> str:
        string = ""
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
        return string
