import numpy as np

def rk4(f, t, y0, dt):

    def step(F, T, Y, DT):
        k1 = DT * f(T, Y)
        k2 = DT * f(T + 0.5*DT, Y + 0.5*k1)
        k3 = DT * F(T + 0.5*DT, Y + 0.5*k2)
        k4 = DT * F(T + DT, Y + k3)

        y_next = Y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        return y_next
    
    t_start, t_end = t
    times = np.arange(t_start, t_end+dt, dt)
    solution = np.zeros((len(times), len(y0)))
    solution[0] = y0

    y = y0.copy()
    for _ in range(len(times)-1):
        y = step(f, times[_], y, dt)
        solution[_+1] = y
    
    return times, solution

def adaptive_rk45():
    pass