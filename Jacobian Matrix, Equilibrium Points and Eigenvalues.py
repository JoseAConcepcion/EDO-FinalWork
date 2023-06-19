from sympy import symbols, Matrix
import numpy as np
import sympy as sp

# Define the variables
x, y, z, r, k, b, a, n, m, u, a1, p, n1 = symbols(
    'x y z r k b a n m u a1 p n1')

# Define the equations
eq1 = r*x*(1-x/k) - b*x - a*x*z
eq2 = b*x - (n*y*z)/(y+m) - u*y
eq3 = a1*x*z + p*z**2 - (n1*z**2)/(y+m)

# Verify if E2 is an equilibrium point
dx = sp.simplify(
    sp.Subs(eq1, (x, y, z), (((k*(r-b))/r), (b*k*(r-b))/(u*r), 0)).doit())
dy = sp.simplify(
    sp.Subs(eq2, (x, y, z), (((k*(r-b))/r), (b*k*(r-b))/(u*r), 0)).doit())
dz = sp.simplify(
    sp.Subs(eq3, (x, y, z), (((k*(r-b))/r), (b*k*(r-b))/(u*r), 0)).doit())
print("E2 is an equilibrium point:", {dx == 0 and dy == 0 and dz == 0})

separator = '- - '*25
print(separator)

# Define the variables as a list
variables = [x, y, z]

# Define the equations as a list
equations = [eq1, eq2, eq3]

# Create the Jacobian matrix
Jacobian = Matrix(equations).jacobian(variables)
Jacobian = sp.simplify(Jacobian)
print("Jacobian matrix:")
print(Jacobian)

print(separator)

# Verify the local stability of the equilibrium point E1
E1 = sp.Subs(Jacobian, (x, y, z), (0, 0, 0)).doit()
E1 = sp.simplify(E1)
eigenvaluesE1 = list(sp.Matrix.eigenvals(E1, simplify=True, multiple=True))
print(f"Jacobian(E1): {E1}")
print(f"Eigenvalues of Jacobian(E1): {eigenvaluesE1}")

print(separator)

# Verify the local stability of the equilibrium point E2
E2 = sp.Subs(Jacobian, (x, y, z), (((k*(r-b))/r), (b*k*(r-b))/(u*r), 0)).doit()
E2 = sp.simplify(E2)
eigenvaluesE2 = list(sp.Matrix.eigenvals(E2).keys())
print(f"Jacobian(E2): {E2}")
print(f"Eigenvalues of Jacobian(E2): {eigenvaluesE2}")

print(separator)


# Checking the Routh–Hurwitz criterion for the E3 stability
A1 = sp.simplify(-Jacobian[0, 0]-Jacobian[1, 1]-Jacobian[2, 2])
print(f"A1: {A1}")

print(separator)

A2 = sp.simplify(-Jacobian[0, 1]*Jacobian[1, 0]-Jacobian[0, 2]*Jacobian[2, 0]-Jacobian[1, 2]
                 * Jacobian[2, 1]+Jacobian[1, 1]*Jacobian[2, 2]+Jacobian[0, 0]*(Jacobian[1, 1]+Jacobian[2, 2]))
print(f"A2: {A2}")

print(separator)

A3 = sp.simplify(Jacobian[0, 2]*(Jacobian[1, 1]*Jacobian[2, 0]-Jacobian[1, 0]*Jacobian[2, 1])+Jacobian[0, 1]*(Jacobian[1, 0]
                 * Jacobian[2, 2]-Jacobian[1, 2]*Jacobian[2, 0])+Jacobian[0, 0]*(Jacobian[1, 2]*Jacobian[2, 1]-Jacobian[1, 1]*Jacobian[2, 2]))
print(f"A3: {A3}")

print(separator)

Hurwitz = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
Hurwitz[0, 0] = A1
Hurwitz[0, 1] = A3
Hurwitz[1, 0] = 1
Hurwitz[1, 1] = A2
Hurwitz[2, 1] = A1
Hurwitz[2, 2] = A3

H1 = A1
# H1Subs=sp.Subs(H1, (x,y,z, r, k, b, a, n, m ,u, a1, p, n1), (2,2,3,0.82,12,0.87, 1.56, 2.41, 0.13, 0.11, 1.12, 1.38, 1.83)).doit()
print(f"H1: {H1}")

print(separator)

H2 = A1*A2-A3
# H2Subs=sp.Subs(H2, (x,y,z, r, k, b, a, n, m ,u, a1, p, n1), (2,2,3,0.82,12,0.87, 1.56, 2.41, 0.13, 0.11, 1.12, 1.38, 1.83)).doit()
print(f"H2: {H2}")

print(separator)

H3 = A1*A2*A3-A3*A3
# H3Subs=sp.Subs(H3, (x,y,z, r, k, b, a, n, m ,u, a1, p, n1), (2,2,3,0.82,12,0.87, 1.56, 2.41, 0.13, 0.11, 1.12, 1.38, 1.83)).doit()
print(f"H3: {H3}")
