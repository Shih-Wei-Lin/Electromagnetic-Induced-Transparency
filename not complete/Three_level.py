import numpy as np
import matplotlib.pyplot as plt

class Density_matrix():
    def __init__(self, n, *args):
        self.state = np.zeros((n,3,3),dtype = complex) #create state for storing data
        
        if args != (): #replace the initial condition
            for i in range(len(args)):
                
                self.state[args[i][0], 0] = args[i][1][0]
                self.state[args[i][0], 1] = args[i][1][1]
                self.state[args[i][0], 2] = args[i][1][2]
                

def rk4(func, t, y, h): #func = bloch(t,sigma) = bloch(t,y), h is dt
    k1 = func(t, y)
    k2 = func(t + h / 2, y + h / 2 * k1)
    k3 = func(t + h / 2, y + h / 2 * k2)
    k4 = func(t + h, y + h * k3)
    
    y_ = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return y_

def Sig(i,j):
    ket = np.zeros((1,3))
    bra = np.zeros((3,1))
    
    ket[0][j] = 1
    bra[i][0] = 1
    return np.dot(bra,ket)
# 
def Master_eq(time,
              gamma_01, gamma_12, gamma_02, 
              Gamma_01, Gamma_12, Gamma_02,
              rabi_omega_12, rabi_omega_02,
              detune_12, detune_02):
    def master(t, rho):
        
        Omega_12 = np.interp(t, time, rabi_omega_12) #insert the number for run code faster
        Omega_02 = np.interp(t, time, rabi_omega_02) #insert the number for run code faster
        
        H = -0.5*np.matrix([[0            ,0            ,Omega_02     ],
                            [0            ,2*detune_12  ,Omega_12     ],
                            [Omega_02     ,Omega_12     ,2*detune_02  ]])
        
        L = Gamma_12*rho[2,2]*(Sig(1,1) - Sig(2,2)) +\
            Gamma_01*rho[1,1]*(Sig(0,0) - Sig(1,1)) +\
            Gamma_02*rho[2,2]*(Sig(0,0) - Sig(2,2)) -\
            gamma_01*rho[0,1]*Sig(0,1) -\
            gamma_02*rho[2,0]*Sig(2,0) -\
            gamma_12*rho[2,1]*Sig(2,1) -\
            gamma_02*rho[0,2]*Sig(0,2) -\
            gamma_12*rho[1,2]*Sig(1,2) -\
            gamma_01*rho[1,0]*Sig(1,0) 
        
        #| rho00,  rho01 ,   rho02 |
        #|                         |
        #| rho10,  rho11 ,   rho12 |
        #|                         |
        #| rho20 , rho21 ,   rho22 |
        
        
        drho = -1j*(np.dot(H,rho) - np.dot(rho,H)) +L
       
        return drho
    return master

def exp_rising(tau,time,t0):
    time = np.linspace(0,time,time+1)
    return np.exp((time-t0)/tau)


if __name__ == "__main__":
    
    Gamma = np.array([0.01,0.01,5]) * 2 * np.pi * 10 ** 6 #[01,12,02]
    gamma = np.array([Gamma[0],Gamma[1]+Gamma[1],Gamma[1]])/ 2 + 0 #[01,12,02]
    
    
    
    detune_02 = 0e6
    detune_12 = 0
    
    Sigma_0 = (0, ((1+0j, 0+0j, 0+0j),
                   (0+0j, 0j, 0+0j),
                   (0+0j, 0+0j, 0+0j)))
    Omega_02 = 3 * 2*np.pi*10 ** 6
    Omega_12 = 0 * 2*np.pi* 10 ** 6
    
    tf = 5 * 10 ** - 6
    N = 5000
    
    t = np.linspace(0, tf, N)
    dt = t[1] - t[0]
    
    
    wave_02 = np.concatenate((np.linspace(0, 0, 1000),    
                                np.linspace(1,1,1000),
                                np.linspace(0, 0, 3000)))
    
    wave_12 = np.concatenate((np.linspace(0, 0, 1000),    
                                np.linspace(1,1,1000),
                                np.linspace(0, 0, 3000)))
    
    rabi_omega_02 = Omega_02 * wave_02
    rabi_omega_12 = Omega_12 * wave_12
    
    
    rho = Density_matrix(N,Sigma_0)
    
    for i in range(N):
        if i == 0:
            continue
        else:
            rho.state[i, :] = rk4(Master_eq(t,
                                              gamma[0], gamma[1], gamma[2], 
                                              Gamma[0], Gamma[1], Gamma[2],
                                              rabi_omega_12, rabi_omega_02,
                                              detune_12, detune_02),
                                 t[i - 1], np.array(rho.state[i - 1, :]), dt)
    
    plt.figure(0)
    plt.plot(t, abs(rho.state[:,2,2]),label = "rho22")
    plt.plot(t, abs(rho.state[:,1,1]),label = "rho11")
    plt.plot(t, abs(rho.state[:,0,0]),label = "rho00")
    plt.plot(t,wave_02,label = "input")
    plt.legend()
    plt.figure(1)
    plt.plot(t, rho.state[:,0,2].imag,label = "rho02")
    plt.plot(t, rho.state[:,0,1].imag,label = "rho01")
    plt.plot(t, rho.state[:,1,2].imag,label = "rho12")
    plt.plot(t,wave_02,label = "input")
    plt.legend()
    plt.show()
