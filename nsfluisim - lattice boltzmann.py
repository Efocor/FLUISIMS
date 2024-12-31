#...... @FECORO, 2022 ......
"""""
Simulador simple basado en el modelo 'Lattice Boltzmann Method' (LBM) para fluidos en 2D.

- Básicamente es un modelo de fluidos basado en la mecánica estadística y la teoría cinética de gases.

El fluido entra por la izquierda y sale por la derecha. El obstáculo cilíndrico se puede mover con el ratón.
Al mover el obstáculo, se puede observar cómo el fluido interactúa con él.

Ahora, el fluido se modela con un gradiente de colores para la densidad y se superpone con vectores de velocidad.
En término científicos, el fluido es una mezcla de partículas que se mueven en una malla regular (D2Q9),
lo anterior implica que cada partícula tiene 9 posibles direcciones de movimiento.

Las limitaciones de este modelo e implementación son:
- No se considera la viscosidad del fluido.
- No se consideran las fuerzas externas.
- No se consideran las condiciones de frontera más complejas.
- No se consideran las condiciones iniciales más realistas.
- No se considera la estabilidad numérica del modelo.

Código por Felipe Alexander Correa Rodríguez.
"""""
#..................................................| Importación de stack
import numpy as np
import pygame
import sys

#..................................................| dimensiones de nuestra malla
#..... dimensiones de nuestra malla
nx, ny = 400, 200  #..... nx: ancho, ny: alto

#..... inicializamos el famoso pygame
pygame.init()
screen = pygame.display.set_mode((nx, ny))
pygame.display.set_caption("NSFLUISIM @Fluidos Lattice Boltzmann | FECORO")
clock = pygame.time.Clock()

#..... nuestro parámetros de modelo
tau = 0.6                #..... tiempo de relajación
niters = 100000          #..... número máximo de iteraciones
rho0 = 1.0               #..... densidad inicial
u0 = 0.05                #..... velocidad inicial (reducida para mayor estabilidad)
obstacle_pos = [100, ny//2]  #..... posición inicial del obstáculo (cilindro)
obstacle_radius = 20         #..... radio del obstáculo

#..... Definición de las direcciones (D2Q9)
c = np.array([
    [0, 0],    #..... 0: reposo
    [1, 0],    #..... 1: este
    [0, 1],    #..... 2: norte
    [-1, 0],   #..... 3: oeste
    [0, -1],   #..... 4: sur
    [1, 1],    #..... 5: noreste
    [-1, 1],   #..... 6: noroeste
    [-1, -1],  #..... 7: suroeste
    [1, -1]    #..... 8: sureste
], dtype=np.int32)

#..... pesos para cada dirección
w = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=np.float32)

#..... velocidad al cuadrado
c_sqr = 1/3

#..................................................| Funciones
#..... funcion para crear el obstáculo
def crear_obstaculo(nx, ny, pos, radius):
    X, Y = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32), indexing='xy')
    dist = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2)
    obstacle = dist <= radius
    return obstacle

#..... función para calcular las distribuciones de equilibrio
def calcular_f_equilibrium(rho, u):
    cu = np.zeros((9, ny, nx), dtype=np.float32)  #..... c_i ⋅ u
    usqr = 1.5 * (u[0]**2 + u[1]**2)          #..... (3/2) u^2
    for i in range(9):
        cu[i] = 3 * (c[i,0]*u[0] + c[i,1]*u[1])
    feq = w[:, np.newaxis, np.newaxis] * rho[np.newaxis, :, :] * (
        1 + cu + 0.5 * (cu ** 2) - usqr[np.newaxis, :, :]
    )
    return feq

#..... inicialización de las distribuciones de equilibrio
def inicializar():
    f = np.zeros((9, ny, nx), dtype=np.float32)               #..... Distribuciones f_i, shape: 9 x ny x nx
    rho = rho0 * np.ones((ny, nx), dtype=np.float32)         #..... Densidad, shape: ny x nx
    u = np.zeros((2, ny, nx), dtype=np.float32)              #..... Velocidad (u_x, u_y), shape: 2 x ny x nx
    u[0, :, :] = u0                                          #..... u_x inicial
    feq = calcular_f_equilibrium(rho, u)                    #..... Distribuciones de equilibrio
    f = feq.copy()
    obstacle = crear_obstaculo(nx, ny, obstacle_pos, obstacle_radius)  #..... shape: ny x nx
    return f, rho, u, obstacle

#..... función de colisión (modelo BGK)
def colision(f, rho, u):
    feq = calcular_f_equilibrium(rho, u)
    f = f - (f - feq) / tau
    #..... manejo de valores no finitos
    f = np.where(np.isfinite(f), f, feq)
    return f

#..... la función de streaming
def streaming(f):
    for i in range(9):
        #..... desplazar en x (axis=2) y en y (axis=1)
        f[i] = np.roll(f[i], shift=c[i,0], axis=1)  #..... desplazar en x (axis=1)
        f[i] = np.roll(f[i], shift=c[i,1], axis=0)  #..... desplazar en y (axis=0)
    return f

#..... función para aplicar condiciones de frontera (bounce-back para obstáculo y paredes)
def aplicar_condiciones_frontera(f, obstacle):
    #..... condiciones de rebote para el obstáculo
    #..... --refleja distribuciones opuestas
    for i in range(9):
        ib = 8 - i
        f[i][obstacle] = f[ib][obstacle]
    
    #..... condiciones de frontera para las paredes
    #..... frontera oeste (inlet)
    f[1, :, 0] = f[3, :, 0]
    f[5, :, 0] = f[7, :, 0]
    f[8, :, 0] = f[6, :, 0]
    
    #..... frontera este (outlet)
    f[3, :, -1] = f[1, :, -1]
    f[7, :, -1] = f[5, :, -1]
    f[6, :, -1] = f[8, :, -1]
    
    return f

#..... función para actualizar las variables macroscópicas
def actualizar_macros(f):
    rho = np.sum(f, axis=0)  #..... densidad, shape: ny x nx
    u = np.zeros((2, ny, nx), dtype=np.float32)  #..... velocidad, shape: 2 x ny x nx
    for i in range(9):
        u[0] += c[i,0] * f[i]
        u[1] += c[i,1] * f[i]
    #..... manejo de posibles divisiones por cero y NaN
    rho = np.where(rho > 1e-8, rho, rho0)
    u[0] /= rho
    u[1] /= rho
    #..... clipping para limitar los valores de velocidad
    u = np.clip(u, -0.3, 0.3)  #..... ajusta según sea necesario
    return rho, u

#..... la función para calcular la viscosidad cinemática
def calcular_viscosidad(tau):
    nu = (tau - 0.5) / 3
    return nu

#..... función para visualizar el campo de densidad y velocidad con gradiente de colores
def visualizar(rho, u, obstacle):
    #..... se normaliza rho para la visualización
    rho_norm = np.clip(rho / rho0, 0, 1)  #..... shape: ny x nx
    
    #..... crea un gradiente de colores (por ejemplo, de azul a rojo, esto para que sea más claro)
    colors = np.zeros((ny, nx, 3), dtype=np.uint8)
    colors[..., 0] = (rho_norm * 255).astype(np.uint8)      #..... Rojo
    colors[..., 1] = (rho_norm * 128).astype(np.uint8)      #..... Verde
    colors[..., 2] = (255 - rho_norm * 255).astype(np.uint8)  #..... Azul
    
    #..... asigna obstáculo en negro
    colors[obstacle] = [0, 0, 0]  #..... shape: ny x nx
    
    #..... convierte a superficie
    surface = pygame.surfarray.make_surface(colors.swapaxes(0,1))
    
    #..... dibujamos la superficie en la pantalla
    screen.blit(surface, (0, 0))
    
    #..... opcional para mí: superponer el vector de velocidad
    skip = 20  #..... intervalo para mostrar vectores
    for i in range(0, ny, skip):
        for j in range(0, nx, skip):
            vel_x = u[0, i, j]
            vel_y = u[1, i, j]
            #..... manejo de NaN y valores extremos
            if not np.isfinite(vel_x) or not np.isfinite(vel_y):
                continue
            end_pos = (j + int(vel_x * 10), i + int(vel_y * 10))
            #..... asegura que las posiciones estén dentro de la pantalla
            if 0 <= end_pos[0] < nx and 0 <= end_pos[1] < ny:
                pygame.draw.line(screen, (255, 0, 0), (j, i), end_pos, 1)
    
    pygame.display.flip()

#..................................................| Llamamiento a todo lo anterior
#..... función principal
def main():
    global obstacle_pos, obstacle_radius
    f, rho, u, obstacle = inicializar()
    nu = calcular_viscosidad(tau)
    print(f"Viscosidad cinemática: {nu}")
    
    dragging = False
    offset = [0, 0]
    
    for it in range(niters):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                dist_mouse = np.sqrt((mouse_x - obstacle_pos[0])**2 + (mouse_y - obstacle_pos[1])**2)
                if dist_mouse <= obstacle_radius:
                    dragging = True
                    offset = [obstacle_pos[0] - mouse_x, obstacle_pos[1] - mouse_y]
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_x, mouse_y = event.pos
                    obstacle_pos[0] = mouse_x + offset[0]
                    obstacle_pos[1] = mouse_y + offset[1]
                    #..... actualiza el obstáculo
                    obstacle = crear_obstaculo(nx, ny, obstacle_pos, obstacle_radius)
        
        #..... paso de colisión
        f = colision(f, rho, u)
        
        #..... paso de streaming
        f = streaming(f)
        
        #..... aplica condiciones de frontera
        f = aplicar_condiciones_frontera(f, obstacle)
        
        #..... actualiza variables macroscópicas
        rho, u = actualizar_macros(f)
        
        #..... fuerza velocidad en la entrada (método de impulso)
        #..... esto asegura que la velocidad de entrada se mantenga constante
        u[0, 0, :] = u0
        u[1, 0, :] = 0.0
        #..... actualiza rho en la entrada basado en las distribuciones conocidas
        rho[0, :] = (f[0, 0, :] + f[1, 0, :] + f[3, 0, :] +
                     2 * (f[2, 0, :] + f[5, 0, :] + f[6, 0, :])) / (1 - u0)
        f = calcular_f_equilibrium(rho, u)
        
        #..... hacemos check y corregimos valores no finitos
        if not np.all(np.isfinite(f)):
            print("Advertencia: Valores no finitos encontrados en f. Reajustando a equilibrium.")
            f = calcular_f_equilibrium(rho, u)
        
        if not np.all(np.isfinite(rho)):
            print("Advertencia: Valores no finitos encontrados en rho. Reajustando a rho0.")
            rho = rho0 * np.ones((ny, nx), dtype=np.float32)
            u = np.zeros((2, ny, nx), dtype=np.float32)
            u[0, :, :] = u0
        
        if not np.all(np.isfinite(u)):
            print("Advertencia: Valores no finitos encontrados en u. Reajustando a u0.")
            u = np.zeros((2, ny, nx), dtype=np.float32)
            u[0, :, :] = u0
        
        #..... visualizamos cada cierto número de iteraciones
        if it % 1 == 0:
            visualizar(rho, u, obstacle)
        
        #..... limita a 60 FPS
        clock.tick(60)

if __name__ == "__main__":
    main()

#..................................................| Fin del código
# Derechos de autor (c) FECORO 2022.
