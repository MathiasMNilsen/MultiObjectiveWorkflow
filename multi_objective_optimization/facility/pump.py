import numpy as np

# Constants
rho = 1000  # kg/m3
g = 9.81    # m/s2

def power_demand_pump(Qreq, Hreq, **kwargs):

    # Maximum number of pumps in parallel and series
    npp_max = kwargs.get('npp_max', 5) # parallel
    nps_max = kwargs.get('nps_max', 5) # series

    # Check if Qreq is iterable
    if hasattr(Qreq, '__iter__'):
        Qreq = np.array(Qreq)
    else:
        Qreq = np.array([Qreq])

    # Check if Hreq is iterable
    if hasattr(Hreq, '__iter__'):
        Hreq = np.array(Hreq)
    else:
        # If Hreq is not iterable, convert it to an array
        Hreq = Hreq*np.ones_like(Qreq)

    # Check if Qreq and Hreq have the same length
    if len(Qreq) != len(Hreq):
        raise ValueError("Qreq and Hreq must have the same length.")

    power = []
    for Qidx, Q in enumerate(Qreq):
        
        candidates = []
        for p in range(npp_max):
            for s in range(nps_max):

                # Number of pumps in parallel and series
                Np = p + 1
                Ns = s + 1

                # Head
                H = system_head(Q, Np, Ns)

                # Power demand
                P = system_power_demand(Q, Np, Ns)
                
                if H >= Hreq[Qidx]:
                    candidates.append(P)
        
        # find best candidate
        power.append(min(candidates))

    return power
    
           

def system_power_demand(Q, Np, Ns):

    # rate and head per pump
    q = Q/Np
    h = pump_head(q)

    # pump efficiency
    eff = pump_eff(q)

    # power demand
    P = (g*rho)*(h*Ns)*(Q/24/3600)/eff

    return P/1e6 # MW

def system_head(Q, Np, Ns):
    return pump_head(Q/Np)*Ns


def pump_eff(q, omega=2500):
    # Sm3/day to m3/s
    q = q/24/3600

    b0 = 1.000e-10
    b1 = 9.641
    b2 = -38.12
    b3 = 5.741e-3
    b4 = -1.629e-7

    return 0.95*(b0 + b1*q + b2*q**2 + b3*omega*q**2 + b4*q*omega**2)


def pump_head(q, omega=2500):
    # Sm3/day to m3/s
    q = q/24/3600

    a0 = 1.823e3
    a1 = -56.41
    a2 = 64.09
    a3 = 2.061e2
    a4 = 1.712e2
    a5 = -1.868e2

    omega_bar_q = 7.068e2 
    omega_bar_2 = 1.942e7
    omega_bar_3 = 8.672e10


    q_bar = 0.1593
    q_bar_3 = 7.922e-3

    sigma_q = 0.08964
    sigma_omega_q = 4.196e2
    sigma_omega_2 = 3.681e6
    sigma_omega_3 = 2.407e10
    sigma_q_3 = 9.304e-3

    term1 = a0 + a1*(q - q_bar)/sigma_q
    term2 = a2*(omega*q - omega_bar_q)/sigma_omega_q
    term3 = a3*(omega**2 - omega_bar_2)/sigma_omega_2
    term4 = a4*(omega**3 - omega_bar_3)/sigma_omega_3
    term5 = a5*(q**3 - q_bar_3)/sigma_q_3

    return term1 + term2 + term3 + term4 + term5

