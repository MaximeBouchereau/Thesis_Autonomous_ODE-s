import math
import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize
import matplotlib.pyplot as plt

# Numerical methods tested

# Parameters

T_simul = 10                 # Time for ODE simulations
N_iter = 10                  # Number of iterations for implicit methods
step_h = [0.001 , 0.1]       # Time step interval for convergence curves
dyn_syst = "Hénon-Heiles"    # Selected dynamical system ("Pendulum" or "Hénon-Heiles")
eps_simul = 1                # Parameter for high oscillations (Hénon-Heiles system)

if dyn_syst == "Pendulum":
    d = 2
    Y0 = np.array([1.5, 0])
if dyn_syst == "Hénon-Heiles":
    d = 4
    Y0 = np.array([0, 0 , 0.1 , 0.1])


print(" ")
print(60*"-")
print(" Numerical methods for ODE's")
print(60*"-")
print(" ")
print(" - Time for ODE simulations:" , T_simul)
print(" - Number of iterations for implicit methods:", N_iter)
print(" - Initial condition for simulations:" , Y0)
print(" - Time step inteval for convergence curves:" , step_h)
print(" - Selected dynamical system:", dyn_syst)
print(" - Parameter for high oscillations (Hénon-Heiles system):", eps_simul)




# Classes and functions

class ODE:
    def f(t,y):
        """Dynamics of the ODE: Pendulum:
        Inputs:
        - t: Float - Time
        - y: Array of shape (d,) - Space variable"""

        y = np.array(y).reshape(d,)
        z = np.zeros_like(y)
        if dyn_syst == "Pendulum":
            z[0] = np.sin(y[1])
            z[1] = - y[0]
        if dyn_syst == "Hénon-Heiles":
            q1, q2, p1, p2 = y[0], y[1], y[2], y[3]
            z = np.array([p1 / eps_simul, p2, -q1 / eps_simul - 2 * q1 * q2, -q2 - q1 ** 2 + (3 / 2) * q2 ** 2])
        return z

    def f_VC(t,y):
        """New funnction after variable change for Hénon-Héiles.
        Inputs:
         - t: Float - Time variable
         - y: Array of shape (d,) - Space variable"""
        tau = t/eps_simul
        z = np.zeros_like(y)
        z[0] = 2 * np.sin(tau) * (y[0] * np.cos(tau) + y[2] * np.sin(tau)) * y[1]
        z[1] = y[3]
        z[2] = -2 * np.cos(tau) * (y[0] * np.cos(tau) + y[2] * np.sin(tau)) * y[1]
        z[3] = -1 * (y[0] * np.cos(tau) + y[2] * np.sin(tau)) ** 2 + (3/2) * y[1] ** 2 - y[1]
        return z

    def H(y):
        """Hamiltonian of the ODE: Pendulum or Hénon-Heiles:
        Inputs:
        - y: Array of shape (d,n) - Space variable"""
        if dyn_syst == "Pendulum":
            z = 0.5*y[0 , :]**2 + (1 - np.cos(y[1 , :]))
        if dyn_syst == "Hénon-Heiles":
            q1, q2, p1, p2 = y[0, :], y[1, :], y[2, :], y[3, :]
            z =  p1 ** 2 / (2 * eps_simul) + p2 ** 2 / 2 + q1 ** 2 / (2 * eps_simul) + q2 ** 2 / 2 + q1 ** 2 * q2 - q2 ** 3 / 2
        return z

    def Exact_solve(T , h):
        """Exact resolution of the ODE by using a very accurate Python Integrator DOP853
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation"""
        Y = solve_ivp(fun = ODE.f_VC , t_span=(0,T) , y0 = Y0 , t_eval=np.arange(0,T,h) , atol = 1e-13 , rtol = 1e-13 , method = "DOP853").y
        TT = np.arange(0,T,h)
        for n in range(np.size(TT)):
            VC = np.array([[np.cos(TT[n] / eps_simul), 0, np.sin(TT[n] / eps_simul), 0], [0, 1, 0, 0],
                           [-np.sin(TT[n] / eps_simul), 0, np.cos(TT[n] / eps_simul), 0], [0, 0, 0, 1]])
            Y[:,n] = VC@Y[:,n]

        return Y

    def Num_solve(T , h , num_meth):
        """Numerical resolution of the ODE.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - num_meth: Str - Name of the numerical method"""

        TT = np.arange(0,T,h)
        YY = np.zeros((d,np.size(TT)))
        YY[:,0] = Y0

        for n in range(np.size(TT)-1):

            def F_iter(y, num_meth , y_D = None , gamma = None):
                """Function which solves the equation for each iteration of an implicit method by Fixed Point iterations.
                Inputs:
                - y: Array of shape (d, ) - Space variable
                - num_meth: Str - Name of the implicit method
                - y_D: Array of shape (d, ) - Initial point of the Numerical flow. Default: None
                - gamma: Float - Coefficient for composition methods. Default: None"""
                if num_meth == "Backward Euler":
                    return YY[:, n] + h * ODE.f(TT[n], y)
                if num_meth == "Symplectic Euler":
                    return YY[:, n] + h * ODE.f(TT[n] , np.array([y[0:d//2] , YY[d//2:d,n]]))
                if num_meth == "MidPoint":
                    return YY[:, n] + h * ODE.f(TT[n], 0.5 * (YY[:, n] + y))
                if num_meth == "Trapezoïdal":
                    return YY[:, n] + 0.5*h * (ODE.f(TT[n] , YY[:, n]) + ODE.f(TT[n] , y))
                if num_meth == "Composition-MidPoint":
                    return y_D + gamma * h * ODE.f(TT[n], 0.5 * (y_D + y))

            if num_meth == "Forward Euler":
                YY[:, n + 1] = YY[:, n] + h * ODE.f(TT[n] , YY[:, n])

            if num_meth == "RK2":
                YY[:, n + 1] = YY[:, n] + h * ODE.f(TT[n], YY[:, n] + (h/2) * ODE.f(TT[n] + h/2 , YY[:, n]))

            if num_meth == "RK4":
                k1 = ODE.f(TT[n] , YY[:, n])
                k2 = ODE.f(TT[n] + h/2 , YY[:, n] + (h/2)*k1)
                k3 = ODE.f(TT[n] + h/2 , YY[:, n] + (h/2)*k2)
                k4 = ODE.f(TT[n] + h , YY[:, n] + h*k3)
                YY[:, n+1] = YY[:, n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            if num_meth == "Fehlberg":

                a = np.array([[1 / 4, 0, 0, 0, 0], [3 / 32, 9 / 32, 0, 0, 0], [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
                     [439 / 216, -8, 3680 / 513, -845 / 4104, 0], [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]])
                b = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
                #b_2 = np.array([16 / 135 , 0 , 6656 / 12825 , 28561 / 56430 , - 9 / 50 , 2 / 55])

                k1 = ODE.f(TT[n], YY[:, n])
                k2 = ODE.f(TT[n], YY[:, n] + h * a[0, 0] * k1)
                k3 = ODE.f(TT[n], YY[:, n] + h * (a[1, 0] * k1 + a[1, 1] * k2))
                k4 = ODE.f(TT[n], YY[:, n] + h * (a[2, 0] * k1 + a[2, 1] * k2 + a[2, 2] * k3))
                k5 = ODE.f(TT[n], YY[:, n] + h * (a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3 + a[3, 3] * k4))
                k6 = ODE.f(TT[n], YY[:, n] + h * (a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4 + a[4, 4] * k5))
                #YY[:, n + 1] = YY[:, n] + h * (b_2[0] * k1 + b_2[1] * k2 + b_2[2] * k3 + b_2[3] * k4 + b_2[4] * k5 + b_2[5] * k6)
                YY[:, n + 1] = YY[:, n] + h * (b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6)

            if num_meth == "DOPRI5":

                a = np.array([[1 / 5, 0, 0, 0, 0 , 0], [3 / 40, 9 / 40 , 0 , 0 , 0 , 0], [44 / 45, -56 / 15, 32 / 9, 0 , 0 , 0],
                     [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0 , 0], [9017 / 3168, - 355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656 , 0] , [35 / 384 , 0 , 500 / 1113 , 125 / 192 , - 2187 / 6784 , 11 / 84]])
                b = np.array([35 / 384 , 0 , 500 / 1113 , 125 / 192 , - 2187 / 6784 , 11 / 84 , 0])
                #b_2 = np.array([16 / 135 , 0 , 6656 / 12825 , 28561 / 56430 , - 9 / 50 , 2 / 55])
                k1 = ODE.f(TT[n], YY[:, n])
                k2 = ODE.f(TT[n], YY[:, n] + h * a[0, 0] * k1)
                k3 = ODE.f(TT[n], YY[:, n] + h * (a[1, 0] * k1 + a[1, 1] * k2))
                k4 = ODE.f(TT[n], YY[:, n] + h * (a[2, 0] * k1 + a[2, 1] * k2 + a[2, 2] * k3))
                k5 = ODE.f(TT[n], YY[:, n] + h * (a[3, 0] * k1 + a[3, 1] * k2 + a[3, 2] * k3 + a[3, 3] * k4))
                k6 = ODE.f(TT[n], YY[:, n] + h * (a[4, 0] * k1 + a[4, 1] * k2 + a[4, 2] * k3 + a[4, 3] * k4 + a[4, 4] * k5))
                k7 = ODE.f(TT[n], YY[:, n] + h * (a[5, 0] * k1 + a[5, 1] * k2 + a[5, 2] * k3 + a[5, 3] * k4 + a[5, 4] * k5 + a[5, 5] * k6))
                #YY[:, n + 1] = YY[:, n] + h * (b_2[0] * k1 + b_2[1] * k2 + b_2[2] * k3 + b_2[3] * k4 + b_2[4] * k5 + b_2[5] * k6)
                YY[:, n + 1] = YY[:, n] + h * (b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6 + b[6] * k7)

            if num_meth == "Backward Euler":
                y_i = YY[:, n]
                for i in range(N_iter):
                    y_i = F_iter(y_i , "Backward Euler")
                YY[:, n+1] = y_i

            if num_meth == "Symplectic Euler":
                y_i = YY[:, n]
                for i in range(N_iter):
                    y_i = F_iter(y_i , "Symplectic Euler")
                YY[:, n+1] = y_i

            if num_meth == "MidPoint":
                y_i = YY[:, n]
                for i in range(N_iter):
                    y_i = F_iter(y_i , "MidPoint")
                YY[:, n+1] = y_i

            if num_meth == "Trapezoïdal":
                y_i = YY[:, n]
                for i in range(N_iter):
                    y_i = F_iter(y_i , "Trapezoïdal")
                YY[:, n+1] = y_i

            if num_meth == "Composition - MidPoint - Order 4":
                p = 2
                gamma_1 = 1/(2 - 2**(1/(p+1)))
                gamma_2 = -2**(1/(p+1))/(2 - 2**(1/(p+1)))
                gammas = [gamma_1 , gamma_2 , gamma_1]
                y_i = YY[:, n]
                y_i_1 = y_i
                for j in range(len(gammas)):
                    for i in range(N_iter):
                        y_i_1 = F_iter(y_i_1, "Composition-MidPoint", y_i, gammas[j])
                    y_i = y_i_1

                YY[:, n + 1] = y_i

            if num_meth == "Composition - MidPoint - Order 6":
                p = 2
                gamma_1_4 = 1 / (2 - 2 ** (1 / (p + 1)))
                gamma_2_4 = -2 ** (1 / (p + 1)) / (2 - 2 ** (1 / (p + 1)))

                p = 4
                gamma_1_6 = 1 / (2 - 2 ** (1 / (p + 1)))
                gamma_2_6 = -2 ** (1 / (p + 1)) / (2 - 2 ** (1 / (p + 1)))

                gammas_order_4 = [gamma_1_4 , gamma_2_4 , gamma_1_4]
                gammas_order_6 = [gamma_1_6*gamma for gamma in gammas_order_4] + [gamma_2_6*gamma for gamma in gammas_order_4] + [gamma_1_6*gamma for gamma in gammas_order_4]
                gammas = gammas_order_6
                y_i = YY[:, n]
                y_i_1 = y_i
                for j in range(len(gammas)):
                    for i in range(N_iter):
                        y_i_1 = F_iter(y_i_1, "Composition-MidPoint", y_i, gammas[j])
                    y_i = y_i_1

                YY[:, n + 1] = y_i_1


            if num_meth == "Composition - MidPoint - Order 8":
                p = 2
                gamma_1_4 = 1 / (2 - 2 ** (1 / (p + 1)))
                gamma_2_4 = -2 ** (1 / (p + 1)) / (2 - 2 ** (1 / (p + 1)))

                p = 4
                gamma_1_6 = 1 / (2 - 2 ** (1 / (p + 1)))
                gamma_2_6 = -2 ** (1 / (p + 1)) / (2 - 2 ** (1 / (p + 1)))

                p = 6
                gamma_1_8 = 1 / (2 - 2 ** (1 / (p + 1)))
                gamma_2_8 = -2 ** (1 / (p + 1)) / (2 - 2 ** (1 / (p + 1)))

                gammas_order_4 = [gamma_1_4 , gamma_2_4 , gamma_1_4]
                gammas_order_6 = [gamma_1_6 * gamma for gamma in gammas_order_4] + [gamma_2_6 * gamma for gamma in gammas_order_4] + [gamma_1_6 * gamma for gamma in gammas_order_4]
                gammas_order_8 = [gamma_1_8 * gamma for gamma in gammas_order_6] + [gamma_2_8 * gamma for gamma in gammas_order_6] + [gamma_1_8 * gamma for gamma in gammas_order_6]
                gammas = gammas_order_8
                y_i = YY[:, n]
                y_i_1 = y_i
                for j in range(len(gammas)):
                    for i in range(N_iter):
                        y_i_1 = F_iter(y_i_1, "Composition-MidPoint", y_i, gammas[j])
                    y_i = y_i_1

                YY[:, n + 1] = y_i_1


        return YY

    def Hamiltonian_Evolution(T , h , save = False):
        """Studies the evolution of Hamiltonian w.r.t. various symplectic methods.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - save: Boolean - Saves the figure or not"""

        #Num_Meths = ["Forward Euler" , "DOPRI5" , "Symplectic Euler" , "MidPoint" , "Composition - MidPoint - Order 4" , "Composition - MidPoint - Order 6"]
        Num_Meths = ["DOPRI5" , "Symplectic Euler" , "MidPoint" , "Composition - MidPoint - Order 4" , "Composition - MidPoint - Order 6"]
        #Colors = ["red" , "magenta" , "darkgreen" , "green" , "yellowgreen" , "gold"]
        Colors = ["magenta" , "darkgreen" , "green" , "yellowgreen" , "gold"]

        TT = np.arange(0 , T , h)
        Y = np.zeros((d+d*len(Num_Meths),np.size(TT)))
        INIT = np.kron(Y0.reshape(d,1) , np.ones(np.size(TT)))

        for k in range(len(Num_Meths)):
            Y[d*k:d*k+d , :] = ODE.Num_solve(T , h , Num_Meths[k])

        Y[-d:, :] = ODE.Exact_solve(T , h)

        plt.figure()
        plt.semilogy()
        for k in range(len(Num_Meths)):
            plt.plot(TT , np.abs(ODE.H(Y[d*k:d*k+d , :]) - ODE.H(INIT)) , color = Colors[k] , label = Num_Meths[k])
        plt.plot(TT , np.abs(ODE.H(Y[-d: , :]) - ODE.H(INIT)) , linestyle = "dashed" , color = "black" , label = "DOP853")
        plt.title("Evolution of Hamiltonian")
        plt.legend()
        plt.grid()
        plt.xlabel("$t$")
        plt.ylabel("$|H(y(t)) - H(y_0)|$")
        if save == True:
            plt.savefig("Hamiltonian_Evolution_T_"+str(T)+"_h_"+str(h)+".pdf")
        plt.show()

class Convergence(ODE):
    def Error(T , h ,num_meth):
        """Computes the relative error between exact solution an numerical approximation of the solution
        w.r.t. a selected numerical method.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - num_meth: Str - Name of the numerical method"""

        YY_exact = ODE.Exact_solve(T , h)
        YY_app = ODE.Num_solve(T , h , num_meth)

        norm_exact = np.max(np.linalg.norm(YY_exact , 2 , axis = 0))
        norm_error = np.max(np.linalg.norm(YY_exact - YY_app, 2 , axis = 0))

        error = norm_error/norm_exact

        return error

    def Curve(T):
        """Plots a curve of convergence w.r.t. various numerical methods
        Inputs:
        - T: Float - Time for ODE simulations"""
        Num_Meths = ["Forward Euler" , "Backward Euler" , "Symplectic Euler" , "RK2" , "MidPoint" , "Trapezoïdal" , "RK4" , "Composition - MidPoint - Order 4" , "DOPRI5" , "Fehlberg" , "Composition - MidPoint - Order 6" , "Composition - MidPoint - Order 8"]
        Colors = ["purple" , "magenta" , "red" , "blue" , "dodgerblue" , "cyan" , "green" , "yellowgreen" , "orange" , "gold" , "grey" , "black"]

        HH = np.exp(np.linspace(np.log(step_h[0]),np.log(step_h[1]),11))
        E = np.zeros((len(Num_Meths),np.size(HH)))

        for k in range(np.size(HH)):
            print(" Loading : h =  {}  \r".format(str(format(HH[k],'.4E'))).rjust(3), end=" ")

            for i in range(len(Num_Meths)):
                E[i,k] = Convergence.Error(T , HH[k] , Num_Meths[i])

        plt.figure()
        for i in range(len(Num_Meths)):
            plt.loglog(HH, E[i,:], "s", color=Colors[i] , label = Num_Meths[i] , markersize = None)
        plt.legend()
        plt.title("Comparaisons between numerical methods")
        plt.xlabel("h")
        plt.ylabel("Rel. Error")
        plt.grid()
        plt.show()


        pass
