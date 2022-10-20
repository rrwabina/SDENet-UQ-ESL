#! /usr/bin/env python3
#
def vcycle ( A, f ):

#*****************************************************************************80
#
## vcycle() performs one v-cycle on the matrix A.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    02 October 2016
#
#  Author:
#
#    Mike Sussman
#
#  Input:
#
#    A(*,*), the matrix.
#
#    f(*), the right hand side.
#
#  Output:
#
#    v(*), the solution of A*v=f.
#
  import numpy as np
  import scipy.linalg as la

  sizeF = np.size ( A, axis = 0 )
#
#  directSize=size for direct inversion
#
  if sizeF < 15:
    v = la.solve(A,f)
    return v
#
#  N1=number of Gauss-Seidel iterations before coarsening
#
  N1 = 5;
  v = np.zeros(sizeF);
  for numGS in range(N1):
    for k in range(sizeF):
      v[k] = (f[k] - np.dot(A[k,0:k], v[0:k]) \
                   -np.dot(A[k,k+1:], v[k+1:]) ) / A[k,k];
# 
#  Construct interpolation operator from next coarser to this mesh
#  next coarser has ((n-1)/2 + 1 ) points
#
  assert ( sizeF%2 == 1 )
  sizeC =  ( sizeF - 1 ) // 2 + 1
  P = np.zeros((sizeF,sizeC));
#
#  Copy these points.
#
  for k in range(sizeC):
    P[2*k,k] = 1;
#
#  Average these points:
#
  for k in range(sizeC-1):
    P[2*k+1,k] = .5;
    P[2*k+1,k+1] = .5;
#
#  compute residual
#
  residual = f - np.dot(A,v)
#
#  project residual onto coarser mesh
#
  residC = np.dot(P.transpose(),residual)
#
#  Find coarser matrix  (sizeC X sizeC)
#
  AC = np.dot(P.transpose(),np.dot(A,P))

  vC = vcycle(AC,residC);
#
# extend to this mesh
#
  v = np.dot(P,vC)
#
#  N2=number of Gauss-Seidel iterations after coarsening
#
  N2 = 5;
  for numGS in range(N2):
    for k in range(sizeF):
      v[k] = (f[k] - np.dot(A[k,0:k], v[0:k]) \
                   - np.dot(A[k,k+1:], v[k+1:]) ) / A[k,k];
  return v

def vcycle_test ( ):

#*****************************************************************************80
#
## vcycle_test() tests vcycle().
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    02 October 2016
#
#  Author:
#
#    Mike Sussman
#
  import numpy as np
  import platform
  import scipy.linalg as la

  print ( '' )
  print ( 'vcycle_test:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  vcycle applies one V-cycle to a matrix.' )

  N = 2**11+1
  x = np.linspace(0,1,N);
  h = x[1]-x[0]
#
#  A is the [-1,2,-1]/h^2 tridiagonal matrix
#
  A = np.diag ( 2.0 * np.ones(N)       ) \
    - np.diag (       np.ones(N-1),  1 ) \
    - np.diag (       np.ones(N-1), -1 )

  A = A / h**2
#
#  The right hand side is a vector of 1's.
#
  f = np.ones ( N, dtype = float )
#
#  UDIRECT is the exact solution, from Gauss elimination.
#
  udirect = la.solve ( A, f )

  u = np.zeros(N) # initial guess

  for iters in range ( 100 ):
    r = f - np.dot(A,u)
    if la.norm(r)/la.norm(f) < 1.e-10:
      print ( 'vcyle_test: Tolerance achieved.' )
      break
    du = vcycle(A, r)
    u += du

    print ( 'step %d, rel error=%e'% \
      (iters, la.norm(u-udirect)/la.norm(udirect) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'vcycle_test:' )
  print ( '  Normal end of execution.' )
  return

def timestamp ( ):

#*****************************************************************************80
#
## timestamp() prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 April 2013
#
#  Author:
#
#    John Burkardt
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

if ( __name__ == '__main__' ):
  timestamp ( )
  vcycle_test ( )
  timestamp ( )

