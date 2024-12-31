#..... FECORO 2022 .....
"""
Simulación simple de fluidos de Jos Stam

Básicamente  es una simulación de fluidos en 2D basada en la técnica de Jos Stam,
se basa en la resolución de ecuaciones de Navier-Stokes para la simulación de fluidos.

De manera simplificada, tenemos una cuadrícula de N x N celdas, donde cada celda
representa una porción de fluido. Cada celda tiene una densidad y velocidad asociada.

La simulación se realiza en pasos de tiempo, donde en cada paso se actualiza la densidad
y velocidad de cada celda en función de la densidad y velocidad de las celdas vecinas.

La simulación se realiza en 3 pasos principales:
1. Difusión: se difunde la densidad y velocidad a través de la cuadrícula.
2. Proyección: se corrige la velocidad para que cumpla con la ecuación de continuidad.
3. Advección: se actualiza la densidad y velocidad en función de la velocidad actual.

La simulación se realiza en un bucle de tiempo, donde en cada paso se realizan los 3 pasos
anteriores. Además, se pueden agregar fuentes de densidad y velocidad en cualquier momento.

Esta es  solo una implementación simple y corta, hecha para matplotlib animation.

Código por Felipe Correa Rodríguez.

"""

#.............................................| 1. Importación de librerías
#.... stack de librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#.... parámetros del dominio
N = 100  #.... resoluciones de la cuadrícula
diff = 0.0  #.... difusividad
visc = 0.0001  #.... viscosidad
dt = 0.1  #.... intervalo de tiempo

#.... funciones para indexar
def IX(x, y):
    return x + y * N

#.... clase para el simulador de fluidos
class Fluid:
    def __init__(self, N, diffusion, viscosity, dt):
        self.N = N
        self.size = N
        self.diff = diffusion
        self.visc = viscosity
        self.dt = dt

        self.s = np.zeros((N, N))
        self.density = np.zeros((N, N))

        self.Vx = np.zeros((N, N))
        self.Vy = np.zeros((N, N))

        self.Vx0 = np.zeros((N, N))
        self.Vy0 = np.zeros((N, N))

    def add_density(self, x, y, amount):
        self.density[y, x] += amount

    def add_velocity(self, x, y, amount_x, amount_y):
        self.Vx[y, x] += amount_x
        self.Vy[y, x] += amount_y

    def step(self):
        N = self.N
        visc = self.visc
        diff = self.diff
        dt = self.dt
        Vx = self.Vx
        Vy = self.Vy
        Vx0 = self.Vx0
        Vy0 = self.Vy0
        s = self.s
        density = self.density

        self.diffuse(1, Vx0, Vx, visc, dt)
        self.diffuse(2, Vy0, Vy, visc, dt)

        self.project(Vx0, Vy0, Vx, Vy)

        self.advect(1, Vx, Vx0, Vx0, Vy0, dt)
        self.advect(2, Vy, Vy0, Vx0, Vy0, dt)

        self.project(Vx, Vy, Vx0, Vy0)

        self.diffuse(0, s, density, diff, dt)
        self.advect(0, density, s, Vx, Vy, dt)

    def diffuse(self, b, x, x0, diff, dt):
        a = dt * diff * (self.N - 2) * (self.N - 2)
        self.lin_solve(b, x, x0, a, 1 + 4 * a)

    def lin_solve(self, b, x, x0, a, c):
        for _ in range(20):
            x[1:-1,1:-1] = (x0[1:-1,1:-1] + a * (x[1:-1,0:-2] + x[1:-1,2:] +
                                                 x[0:-2,1:-1] + x[2:,1:-1])) / c
            self.set_bnd(b, x)

    def project(self, velocX, velocY, p, div):
        div[1:-1,1:-1] = -0.5 * (velocX[1:-1,2:] - velocX[1:-1,0:-2] +
                                   velocY[2:,1:-1] - velocY[0:-2,1:-1]) / self.N
        p.fill(0)
        self.lin_solve(0, p, div, 1, 4)
        velocX[1:-1,1:-1] -= 0.5 * (p[1:-1,2:] - p[1:-1,0:-2]) * self.N
        velocY[1:-1,1:-1] -= 0.5 * (p[2:,1:-1] - p[0:-2,1:-1]) * self.N
        self.set_bnd(1, velocX)
        self.set_bnd(2, velocY)

    def advect(self, b, d, d0, velocX, velocY, dt):
        Nfloat = float(self.N)
        dt0 = dt * self.N
        X, Y = np.meshgrid(np.arange(self.N), np.arange(self.N))
        x = X - dt0 * velocX
        y = Y - dt0 * velocY

        x = np.clip(x, 0.5, Nfloat + 0.5)
        y = np.clip(y, 0.5, Nfloat + 0.5)

        i0 = np.floor(x).astype(int)
        i1 = i0 + 1
        j0 = np.floor(y).astype(int)
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1

        i0 = np.clip(i0, 0, self.N -1)
        i1 = np.clip(i1, 0, self.N -1)
        j0 = np.clip(j0, 0, self.N -1)
        j1 = np.clip(j1, 0, self.N -1)

        d[1:-1,1:-1] = (s0[1:-1,1:-1] * (t0[1:-1,1:-1] * d0[j0[1:-1,1:-1], i0[1:-1,1:-1]] +
                                            t1[1:-1,1:-1] * d0[j1[1:-1,1:-1], i0[1:-1,1:-1]]) +
                           s1[1:-1,1:-1] * (t0[1:-1,1:-1] * d0[j0[1:-1,1:-1], i1[1:-1,1:-1]] +
                                            t1[1:-1,1:-1] * d0[j1[1:-1,1:-1], i1[1:-1,1:-1]]))
        self.set_bnd(b, d)

    def set_bnd(self, b, x):
        x[0,1:-1] = -x[1,1:-1] if b == 2 else x[1,1:-1]
        x[-1,1:-1] = -x[-2,1:-1] if b == 2 else x[-2,1:-1]
        x[1:-1,0] = -x[1:-1,1] if b == 1 else x[1:-1,1]
        x[1:-1,-1] = -x[1:-1,-2] if b == 1 else x[1:-1,-2]

        x[0,0] = 0.5 * (x[1,0] + x[0,1])
        x[0,-1] = 0.5 * (x[1,-1] + x[0,-2])
        x[-1,0] = 0.5 * (x[-2,0] + x[-1,1])
        x[-1,-1] = 0.5 * (x[-2,-1] + x[-1,-2])

#.... configuración de la simulación
fluid = Fluid(N, diff, visc, dt)

fig, ax = plt.subplots()
im = ax.imshow(fluid.density, cmap='inferno', origin='lower')
plt.colorbar(im)

#.............................................| 2. Animación de la simulación
def actualiza(frame):
    #.... agregar algunas fuentes de densidad y velocidad
    if frame == 0:
        fluid.add_density(N//2, N//2, 100)
        fluid.add_velocity(N//2, N//2, 0, 5)
    fluid.step()
    im.set_data(fluid.density)
    return [im]

ani = animation.FuncAnimation(fig, actualiza, frames=200, interval=50, blit=True)
plt.show()
