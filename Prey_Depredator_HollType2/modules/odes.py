class Prey_Depredator_HollingTypeII:

    # # Parametros
    # r = 0
    # beta = 0
    # alpha = 0
    # alpha1 = 0
    # eta = 0
    # eta1 = 0
    # k = 0
    # rho = 0
    # m = 0
    # mu = 0

    # def __init__(self, params):
    #     # [r, alpha, beta, rho, m, mu, eta, alpha1, eta1, k]
    #     # lo ideal era hacer el vector con los parametros pero es ilegible en las formulas
    #     self.r = params[0]
    #     self.beta = params[1]
    #     self.alpha = params[2]
    #     self.alpha1 = params[3]
    #     self.eta = params[4]
    #     self.eta1 = params[5]
    #     self.k = params[6]
    #     self.rho = params[7]
    #     self.m = params[8]
    #     self.mu = params[9]
        
    # # EDO
    # def f1(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.r*x * (1 - x/self.k) - self.beta*x - self.alpha*x*z

    # def f2(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.beta*x - self.eta*y*z/(self.m + y) - self.mu*y

    # def f3(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.alpha1*x*z - self.rho*z*z - self.eta1*y*z/(self.m + y)

    # def __str__(self) -> str:
    #     string = "Parametros\n"
    #     string += "r = " + str(self.r) + "\n"
    #     string += "alpha = " + str(self.alpha) + "\n"
    #     string += "beta = " + str(self.beta) + "\n"
    #     string += "rho = " + str(self.rho) + "\n"
    #     string += "m = " + str(self.m) + "\n"
    #     string += "mu = " + str(self.mu) + "\n"
    #     string += "eta = " + str(self.eta) + "\n"
    #     string += "alpha1 = " + str(self.alpha1) + "\n"
    #     string += "eta1 = " + str(self.eta1) + "\n"
    #     string += "k = " + str(self.k) + "\n"

    #     string += "\nEDO\n"
    #     string += "dx/dt = " + "{0}*x*(1-x/{1}) - {2}*x - {3}*x*z".format(self.r,self.k,self.beta,self.alpha) + "\n"
    #     string += "dy/dt = " + "{0}*x - {1}*y*z/({2} + y) - {3}*y".format(self.beta,self.rho,self.rho,self.mu) + "\n"
    #     string += "dz/dt = " + "{0}*x*z - {1}*z^2 - {2}*y*z/(y + {3})".format(self.alpha1,self.rho,self.eta1, self.m) + "\n"

    #     return string
    # r = 0
    # beta = 0
    # alpha = 0
    # alpha1 = 0
    # eta = 0
    # eta1 = 0
    # k = 0
    # rho = 0
    # m = 0
    # mu = 0

    # def __init__(self, params):
    #     # [r, alpha, beta, rho, m, mu, eta, alpha1, eta1, k]
    #     # lo ideal era hacer el vector con los parametros pero es ilegible en las formulas
    #     self.r = params[0]
    #     self.beta = params[1]
    #     self.alpha = params[2]
    #     self.alpha1 = params[3]
    #     self.eta = params[4]
    #     self.eta1 = params[5]
    #     self.k = params[6]
    #     self.rho = params[7]
    #     self.m = params[8]
    #     self.mu = params[9]

    # # EDO
    # def f1(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.r*x * (1 - x/self.k) - self.beta*x - self.alpha*x*z

    # def f2(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.beta*x - self.eta*y*z/(self.m + y) - self.mu*y

    # def f3(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.alpha1*x*z - self.rho*z*z - self.eta1*y*z/(self.m + y)

    # def __str__(self) -> str:
    #     string = "Parametros\n"
    #     string += "r = " + str(self.r) + "\n"
    #     string += "alpha = " + str(self.alpha) + "\n"
    #     string += "beta = " + str(self.beta) + "\n"
    #     string += "rho = " + str(self.rho) + "\n"
    #     string += "m = " + str(self.m) + "\n"
    #     string += "mu = " + str(self.mu) + "\n"
    #     string += "eta = " + str(self.eta) + "\n"
    #     string += "alpha1 = " + str(self.alpha1) + "\n"
    #     string += "eta1 = " + str(self.eta1) + "\n"
    #     string += "k = " + str(self.k) + "\n"

    #     string += "\nEDO\n"
    #     string += "dx/dt = " + "{0}*x*(1-x/{1}) - {2}*x - {3}*x*z".format(self.r,self.k,self.beta,self.alpha) + "\n"
    #     string += "dy/dt = " + "{0}*x - {1}*y*z/({2} + y) - {3}*y".format(self.beta,self.rho,self.rho,self.mu) + "\n"
    #     string += "dz/dt = " + "{0}*x*z - {1}*z^2 - {2}*y*z/(y + {3})".format(self.alpha1,self.rho,self.eta1, self.m) + "\n"

    #     return string
    
    # Parametros new

    a1 = 0
    a2 = 0
    b2 = 0
    c = 0
    d = 0
    k = 0
    sigma = 0
    ro = 0
    beta = 0

    def __init__(self, params): 
        self.a1 = params[0]
        self.a2 = params[1]
        self.b2 = params[2]
        self.c = params[3]
        self.d = params[4]
        self.k = params[5]
        self.sigma = params[6]
        self.ro = params[7]
        self.beta = params[8]



    # EDO
    # def f1(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.ro*x * (1 - x/self.k) - self.a1*x*y

    # def f2(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.c*self.a1*x*y - self.d*y - ((self.a2*y*z)/(y + self.b2))

    # def f3(self, t: float, x: float, y: float, z: float) -> float:
    #     return self.sigma*z*z - ((self.beta*z*z)/(y + self.b2))

    # EDO
    def f1(self, t: float, x: float, y: float, z: float) -> float:
        return 2*x - 2*y + 3*z

    def f2(self, t: float, x: float, y: float, z: float) -> float:
        return 1*x + y + 1*z

    def f3(self, t: float, x: float, y: float, z: float) -> float:
        return x + 3*y - z 
