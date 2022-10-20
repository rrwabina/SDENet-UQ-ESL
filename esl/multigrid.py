import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import timeit

def func(x,y):
	return -2.0*(np.pi**2)*np.sin(np.pi*(x))*np.sin(np.pi*(y))


def solution(dims):
	h_x = 1.0/(dims[0] + 1)
	h_y = 1.0/(dims[1] + 1)
	u = np.zeros((dims[0] + 2, dims[1] + 2))
	for x_i in range(1, dims[0]+1):
		for y_j in range(1, dims[1]+1):
			u[x_i, y_j] = np.sin(np.pi*h_x*x_i)*np.sin(np.pi*h_y*y_j)
	#b[1:-1, 1:-1] = 1.0
	return u

"""
This is a function to populate a matrix that acts as the right hand function evaluated over a discrete grid. 
Inputs to this function is a list of grid point sizes, i.e. [Nx, Ny] and a pointer to mathematical function.
The put is a matrix b.
"""
def create_B(dims, func):
	h_x = 1.0/(dims[0] + 1)
	h_y = 1.0/(dims[1] + 1)

	b = np.zeros((dims[0]+2, dims[1]+2))

	for x_i in range(1, dims[0]+1):
		for y_j in range(1, dims[1]+1):
			b[x_i, y_j] = func(h_x*x_i, h_y*y_j)

	return b

"""
Smoothing function used by multigrid method. The smoothing is done by applying N Gauss-Seidel iteration.
The inputs:
	T - a matrix of Nx+2, Ny+2 elements that is used to approximate the solution of the partial differential equation. 
	b - right hand side of the partial differential equation.
	dims - a list of grid sizes.
	steps - the number of smoothing iterations
Outputs:
	Returns an updated version of T
"""
def smooth(T, b, dims, steps):
	h_x = 1.0/(dims[0] + 1)
	h_y = 1.0/(dims[1] + 1)
	N = dims[0]*dims[1]

	for k in range(steps):
		
		for y_j in range(1, dims[1]+1):
			for x_i in range(1, dims[0] + 1):
				T[x_i, y_j] = ( (T[x_i-1, y_j] + T[x_i+1,y_j])/(h_x**2) + (T[x_i, y_j+1] + T[x_i, y_j-1])/(h_y**2) - (b[x_i,y_j]))/(2.0/(h_x**2) + 2.0/(h_y**2))

	return T

"""
Residual function used by multigrid method and it gives an quantified measure of how close is the numerical solution is to the exact solution. When the resiudal is zero, it means that T solve the system of algebraic equations that approximates the partial differential equation. 
The inputs:
	T - a matrix of Nx+2, Ny+2 elements that is used to approximate the solution of the partial differential equation. 
	b - right hand side of the partial differential equation.
	dims - a list of grid sizes.
Outputs:
	y - residual
	res_temp - norm of the residual
"""
def residual(T, b, dims):
	h_x = 1.0/(dims[0] + 1)
	h_y = 1.0/(dims[1] + 1)
	y = np.zeros((dims[0]+2, dims[1]+2))
	 
	sum_temp = 0.0
	for x_i in range(1, dims[0]+1):
		for y_j in range(1, dims[1]+1):
			y[x_i, y_j] = -(T[x_i-1, y_j] - 2*T[x_i,y_j] + T[x_i+1, y_j])/(h_x**2) - (T[x_i, y_j-1] - 2*T[x_i,y_j] + T[x_i, y_j+1])/(h_y**2) + b[x_i,y_j]
			sum_temp += y[x_i, y_j]**2;

	res_temp = np.sqrt(sum_temp)

	return y, res_temp


"""
restriction function which maps the matrix on a fine grid to a coarser grid
The inputs:
	T - a matrix of Nx+2, Ny+2 elements that is used to approximate the solution of the partial differential equation on a fine grid.
	dims - a list of grid sizes.
Outputs:
	E - projection of T onto a coarser grid.
"""
def restriction(T, dims):
	E = np.zeros((dims[0]+2, dims[1]+2))
	for i in range(1,dims[0]+1):
		for j in range(1, dims[1]+1):
			E[i,j] = T[2*(i), 2*j ]

	return E

"""
Interpolation function which maps matrices on coarse grids to a finer ones. It uses bi-linear interpolation to acheive its goal.
The inputs:
	T - a matrix of Nx+2, Ny+2 elements that is used to approximate the solution of the partial differential equation on a coarse grid.
	dims - a list of grid sizes.
Outputs:
	E - projection of T onto a finer grid.
"""
def interpolation(T, dims):
	E = np.zeros((dims[0]+2, dims[1]+2))
	Nx_2 = int(np.floor(dims[0]/2))
	Ny_2 = int(np.floor(dims[1]/2))

	for i in range(0, Nx_2+ 1):
		for j in range(0,Ny_2+1):
			E[2*i, 2*j] = T[i,j]
			E[2*i, 2*j + 1] = (T[i,j] + T[i,j+1])/2.0
			E[2*i + 1, 2*j] = (T[i+1,j] + T[i,j])/2.0
			E[2*i + 1, 2*j+ 1] = (T[i,j] + T[i,j+1] + T[i+1,j] + T[i+1,j+1])/4.0

	return E

"""
V-Cycle function which is the core of multigrid method. This is a recursive function that applies smoothing to the error mapped to different coearser grids and then interpolated and used as correction term.
The inputs:
	T - a matrix of Nx+2, Ny+2 elements that is used to approximate the solution of the partial differential equation on the finest grid. 
	b - the right hand side of the equation
	dims - a list of grid sizes.
	steps - a list with the number of pre-smoothing and post-smoothing iterations
Outputs:
	T - the updated value of T
	res_norm - norm of the residual.
"""
def vCycle(T, b, dims, steps):
	N = dims[0]*dims[1]
	T = smooth(T,b,dims, steps[0])
	if dims[0] >= 3 and dims[1] >= 3:
		res, _ = residual(T,b,dims)
		dims_s = [int(np.floor(dims[0]/2.0)),int(np.floor(dims[1]/2.0))]
		res_s = restriction(res, dims_s)
		T_2 = np.zeros((dims_s[0] + 2, dims_s[1] + 2))
		T_2, _ = vCycle(T_2, res_s, dims_s, steps )
		T_3 = interpolation(T_2, dims)
		T = T + T_3
	T = smooth(T,b,dims, steps[1])

	res, res_norm = residual(T,b,dims)
	res_norm = (1.0/N)*np.sqrt(res_norm)

	return T, res_norm