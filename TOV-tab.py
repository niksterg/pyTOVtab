
# coding: utf-8

# 
# **TOV Stars with tabulated equation of state**
# 
# N. Stergioulas
# 
# Aristotle University of Thessaloniki
# 
# v1.0 (June 2018)
# 
###### Content provided under a Creative Commons Attribution license, 
# [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); 
# code under [GNU GPLv3 License](https://choosealicense.com/licenses/gpl-3.0/). 
# (c)2018 [Nikolaos Stergioulas](http://www.astro.auth.gr/~niksterg/)


from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import scipy as sp
from scipy import integrate
from scipy import optimize
from scipy.interpolate import PchipInterpolator
import sys
from decimal import Decimal
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from IPython.display import Image
from IPython.display import clear_output, display

import os
import contextlib

# these functions are used to suppress lsoda writing to warnings to stdout

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

# constants in CGS
mb = 1.66e-24
c = 2.9979e10
G=6.67408e-8
Msun=1.989e33
# scales
Length = G*Msun/c**2
Time = Length/c
Density = Msun/Length**3

# read command-line arguments

import argparse

parser = argparse.ArgumentParser(description='TOV solution for Piecewise Polytropic EOS')

parser.add_argument('rho_c', type=float, default=1e15,
                    help='central density')

args = parser.parse_args()

rho_c = args.rho_c/Density


# load eos file
eos_name = 'FPS_PP_HnG.dat'
#eos_name = 'FPS_GPP.dat'

#eos_name = 'EOS-RNS1/FPS_PP.dat'
N_eos = int(open(eos_name).readline().rstrip())

e_over_c2_CGS, P_CGS, H_c2_CGS, n_CGS = np.loadtxt(eos_name, skiprows=1, unpack=True)

# make dimensionless

epsilon = e_over_c2_CGS/Density
pressure = P_CGS/(Density*c**2)
enthalpy = H_c2_CGS/c**2
density = mb*n_CGS*Length**3/Msun

  
log_epsilon = np.log10(epsilon)
log_pressure = np.log10(pressure)
log_density = np.log10(density)

log_pressure_of_log_density = PchipInterpolator(log_density,log_pressure)
log_density_of_log_pressure = PchipInterpolator(log_pressure,log_density)
log_epsilon_of_log_density = PchipInterpolator(log_density,log_epsilon)
log_epsilon_of_log_pressure = PchipInterpolator(log_pressure,log_epsilon)

  
eps_c = pow(10.0,log_epsilon_of_log_density(np.log10(rho_c)))
P_c = pow(10.0,log_pressure_of_log_density(np.log10(rho_c)))

 
# ## Define the system of ODEs to be solved

 

def f(r, y):
    
    eps = pow(10.0,log_epsilon_of_log_pressure(np.log10(y[0])))
    
    return [ -( eps + y[0] )*( y[1] + 4.0*np.pi*pow(r,3.0)*y[0] )/( r*(r-2.0*y[1]) ), 
            
             4*np.pi*pow(r,2.0)*eps,
            
             2.0*( y[1] + 4.0*np.pi*pow(r,3.0)*y[0] )/( r*(r-2.0*y[1]) ) 
           ]


# ## Set the central value of $\nu_c$ and the starting values for the system of ODEs:

# set an arbitrary starting value for nu at the center
nu_c = -1.0

# set a safe max r, based on 4x radius of 1 Msun uniform density Newt. model
r_max = 4.0 * pow( 3.0/(4.0*np.pi*eps_c), 1.0/3.0)

# create an equidistant array of values for r
Npoints = (51, 101, 201, 401, 801, 1601,  3201, 6401, 12801, 25601, 51201)
#N = 25601
N = 801
r = np.linspace(0.0, r_max, N)
dr = r[1] - r[0]

# compute P, m, nu at r=dr by Taylor expansion
P_1 =  P_c - (2.0*np.pi)*(eps_c+P_c)*(P_c+(1.0/3.0)*eps_c)*pow(dr,2.0)
m_1 =  (4.0/3.0)*np.pi*eps_c*pow(dr, 3.0)
nu_1 = nu_c + 4.0*np.pi*(P_c+(1.0/3.0)*eps_c)*pow(dr,2.0)

# set starting values at r=dr for numerical integration
y0 = [P_1, m_1, nu_1]


# ## Numerical solution

# Define an instant of the numerical solution of the ODE system

solve = integrate.ode(f)
solve.set_integrator('lsoda', rtol=1e-12, atol=1e-50,ixpr=True);
solve.set_initial_value(y0, dr);
solve.set_f_params();


# Integrate from starting point to the surface (where $P=0$):

# create the solution vector
y = np.zeros((len(r), len(y0)))

# fill the solution vector with the values at the center
y[0,:] = [P_c, 0.0, nu_c]

# initialize counter
idx = 1

# integrate repeatedly to next grid point until P becomes zero
while solve.successful() and solve.t < r[-1] and solve.y[0]>0.0:
    
      y[idx, :] = solve.y
      solve.integrate(solve.t + dr)
      idx += 1

# last grid point with positive pressure
idxlast = idx-1 

# radius at last positive pressure grid point
R_last = r[idxlast]

# mass at last positive pressure grid point
Mass_last = y[idxlast][1]


display(R_last, Mass_last)


# Locate real radius by finding the location where h=1.0.

# use last 4 points to construct interpolant
r_data = np.zeros(4)
h_data = np.zeros(4)
eps_data = np.zeros(4)
rho_data = np.zeros(4)
P_data = np.zeros(4)
dmdr_data = np.zeros(4)

for i in range(idxlast-3,idxlast+1):
    r_data[i-idxlast+3] = r[i]
    eps_data[i-idxlast+3] = pow(10.0,log_epsilon_of_log_pressure(np.log10(y[i][0])))
    rho_data[i-idxlast+3] = pow(10.0,log_density_of_log_pressure(np.log10(y[i][0])))
    P_data[i-idxlast+3] = y[i][0]
    h_data[i-idxlast+3] = (eps_data[i-idxlast+3] + P_data[i-idxlast+3])                            / rho_data[i-idxlast+3] -1.0
    dmdr_data[i-idxlast+3] = 4.0*np.pi*r[i]**2*eps_data[i-idxlast+3]

h_interp = PchipInterpolator(r_data, h_data)

# find the root using Brent's method
#Radius = optimize.brentq( h_interp, r_data[0], r_data[3]+3*dr, xtol=1e-16 )

#display(Radius)


# Locate radius more accurately (to 4th-order) using a cubic Hermite interpolant of the specific enthalpy h-1.


def hHerm (r):
    r_last_1 = R_last-dr
    r_last = R_last
    w = (r-r_data[2])/dr
    m_last_1 = y[idxlast-1][1]
    m_last = y[idxlast][1]
    dhdr_last_1 = - (h_data[2]+1.0)*(m_last_1 +                             4.0*np.pi*r_last_1**3*y[idxlast-1][0])/                            (r_last_1*(r_last_1-2.0*m_last_1))
    dhdr_last = - (h_data[3]+1.0)*(m_last +                            4.0*np.pi*r_last**3*y[idxlast][0])/                            (r_last*(r_last-2.0*m_last))
    return (h_data[2]+1.0)*(2.0*pow(w,3.0)-3.0*pow(w,2.0)+1.0)+                           (h_data[3]+1.0)*(2.0*pow(1.0-w,3.0)-3.0*pow(1.0-w,2.0)+1.0)                           + ( dhdr_last_1*(pow(w,3.0)-2.0*pow(w,2.0)+w) -                            dhdr_last*(pow(1-w,3.0)-2.0*pow(1-w,2.0)+1-w))*dr -1.0



#Radius = optimize.brentq( hHerm, r_data[0], r_data[3]+3*dr, xtol=1e-16 )
#display(Radius)

# temporarily accept R_last as the radius
Radius = R_last

# Correct mass by adding last missing piece by Simpson's rule (finding an intemediate point by pchip interpolation):


dmdr_interp_pchip = PchipInterpolator(r_data, dmdr_data)
dmdr_midpoint = dmdr_interp_pchip((R_last+Radius)/2)
Dmass_simps = (1.0/3.0)*(Radius-R_last)/2*(dmdr_interp_pchip(R_last)+4.0*dmdr_midpoint+dmdr_interp_pchip(Radius))

#Mass = Mass_last + Dmass_simps

# temporarily accept Mass_last as the mass
Mass = Mass_last 

#display(Mass)


# Construct table with main solution variables:


values = np.zeros((idxlast+1, 10)) 

for i in range(0,idxlast+1): 
    values[i][0] = r[i]
    values[i][1] = pow(10.0,log_density_of_log_pressure(np.log10(y[i][0]))) # rho
    values[i][2] = pow(10.0,log_epsilon_of_log_pressure(np.log10(y[i][0]))) # epsilon
    values[i][3] = y[i][0]   # P
    values[i][4] = y[i][1]   # m
    values[i][5] = y[i][2]   # nu (arbitrary)

values[0][6] = 0.0
for i in range(1,idxlast+1):     
    values[i][6] = - np.log(1.0-2.0*y[i][1]/r[i])   # lambda
    
values[:, 7] = (values[:, 2] + values[:, 3])/values[:, 1]  # h

values[:, 8] = - (values[:, 4] + 4.0*np.pi*pow(values[:, 0], 3.0)*values[:, 3])/                      ( values[:, 0]*(values[:, 0] - 2.0*values[:, 4]))
                    # (e+P)^{-1} dP/dr directly from rhs of TOV eqn
        
values[0][8] = 0.0   # fix value at the center 

#values[:, 9] = (values[:,2]+values[:,3])/values[:,3]*np.gradient(values[:,3],values[:,2],edge_order=2)

values[:, 9] = values[:,1]/values[:,3]*np.gradient(values[:,3],values[:,1],edge_order=2)

values[idxlast, 9] = values[idxlast-1, 9] # fix value at the surface


# Match $\nu$ at the surface, using Schwarzshild vacuum solution:

# arbitrary nu at the surface
nu_s_old = y[idxlast][2]

# correct nu at the surface
nu_s = np.log(1.0-2.0*Mass/Radius)

# shift nu inside star by difference
values[:, 5] = values[:, 5] + (-nu_s_old + nu_s)


# Compute baryon mass and alternative expression for gravitational mass:

# construct radius array and integrands for baryon and alternative mass integration

rint = np.zeros(idxlast+1)
m0int = np.zeros(idxlast+1)
mint_alt = np.zeros(idxlast+1)

# fill radius array and integrands 

for i in range(0,idxlast+1): 
    rint[i] = values[i][0]
    m0int[i] = 4.0*np.pi*pow(rint[i],2.0)*np.exp(values[i][6]/2.0)*values[i][1]
    mint_alt[i] = 4.0*np.pi*pow(rint[i],2.0)*np.exp((values[i][5]+values[i][6])/2.0)                   *(values[i][2]+3.0*values[i][3])

# integrate using Simpson's method
M0_last = integrate.simps( m0int, dx=dr)
M_alt_last = integrate.simps( mint_alt, dx=dr, even='last')

# correct M0 and M_alt by adding last trapezoid
M0 = M0_last + 0.5*4.0*np.pi*R_last**2*np.exp(values[idxlast][6]/2.0)                       *values[idxlast][1]*(Radius-R_last)

M_alt = M_alt_last + 0.5*4.0*np.pi*R_last**2* np.exp((values[idxlast][5]
                            +values[idxlast][6])/2.0)*(values[idxlast][2] \
                                +3.0*values[idxlast][3]) *(Radius-R_last)

# compute relative difference between mass and alt. mass
M_reldiff = (Mass-M_alt)/Mass


# # Main results

N_gridpoints = idxlast+1

# print('Number of grid points =', N_gridpoints)
# print('rho_c =', rho_c)
# print('epsilon_c =', eps_c)
# print('P_c =', P_c)
# print('dr =', dr)
# print('Radius of last grid point =', R_last)
# print('Extrapolated Radius at zero pressure =', '%.16f'% Radius)
# print('Baryon Mass =', M0)
# print('Gravitational Mass =', '%.16f'% Mass)
# print('Alternative Mass =', M_alt)
# print('Rel. diff. in Mass =', M_reldiff)


 
# # Convert to CGS

 

c=2.9979e10
G=6.67408e-8
Msun=1.989e33
Length = G*Msun/c**2
Time = Length/c
Density = Msun/Length**3
dr_CGS = dr*Length
# print('Number of grid points =', N_gridpoints)
# print('rho_c =', rho_c*Density)
# print('epsilon_c/c^2 =', eps_c*Density)
# print('epsilon_c =', eps_c*Density*c**2)
# print('P_c =', P_c*Density*c**2)
# print('dr =', dr_CGS)
# print('Radius of last grid point =', R_last*Length)
# print('Extrapolated Radius at zero pressure =', Radius*Length)
# print('Baryon Mass =', M0*Msun)
# print('Gravitational Mass =', Mass*Msun)
# print('Alternative Mass =', M_alt*Msun)
# print('Rel. diff. in Mass =', M_reldiff)

print('Number of grid points =', N_gridpoints)
print('rho_c =', rho_c*Density)
print('epsilon_c/c^2 =', eps_c*Density)
print('Radius of last grid point (km) =', R_last*Length/1e5)
print('Baryon Mass (Msun) =', M0)
print('Gravitational Mass (Msun) =', Mass)

values_CGS = np.zeros((idxlast+1, 10)) 

values_CGS[:, 0] = values[:, 0] * Length
values_CGS[:, 1] = values[:, 1] * Density  # rho
values_CGS[:, 2] = values[:, 2] * Density*c**2  # epsilon
values_CGS[:, 3] = values[:, 3] * Density*c**2  # P
values_CGS[:, 4] = values[:, 4] * Msun  # m
values_CGS[:, 5] = values[:, 5]         # nu
values_CGS[:, 6] = values[:, 6]         # lambda
values_CGS[:, 7] = values[:, 7] * c**2  # h
values_CGS[:, 8] = values[:, 8] / Length   # (epsilon+P)^{-1} dP/dr
values_CGS[:, 9] = values[:, 9]         # Gamma


# # Write output files


np.savetxt('TOV_output.dat', values)
np.savetxt('TOV_output_CGS.dat', values_CGS)


