#.... @FECORO, 2022 ....

"""
Simulador de fluidos avanzado basado en las ecuaciones de Navier-Stokes.

Este script implementa un simulador de fluidos avanzado basado en las ecuaciones de Navier-Stokes
utilizando el método de proyección de presión. El simulador permite ajustar los parámetros de
difusividad, viscosidad y el intervalo de tiempo, y visualizar la densidad y los vectores de velocidad
del fluido en tiempo real. También se pueden agregar densidad y velocidad al fluido haciendo clic y
arrastrando en el sub-canvas.

Controles:
- Haz clic y arrastra en el sub-canvas para agregar densidad y velocidad.
- Ajusta los parámetros de difusividad, viscosidad y Δt utilizando los sliders.
- Presiona 'Iniciar' para comenzar la simulación.
- Presiona 'Pausar' para detener/reanudar la simulación.
- Presiona 'Reiniciar' para limpiar la simulación. (Debes iniciar o colocar puntos antes de ver cambios)
- Cambia el modo de visualización utilizando los botones correspondientes.

Autor: Felipe Alexander Correa Rodríguez
Fecha: 2022-04-21

"""
#..... stack de librerías

import pygame
import numpy as np
import sys
import math
import os
from pygame.locals import *
import time

#.........................................................| 1. Configuración inicial de pygame
#..... configuración inicial de pygame
pygame.init()

#..... dimensiones de la ventana principal
WIDTH, HEIGHT = 1280, 720
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NSFLUISIM @Fluidos Navier-Stokes | FECORO")

#..... colores
negro = (0, 0, 0)
blanco = (255, 255, 255)
azul = (0, 0, 255)
rojo = (255, 0, 0)
verde = (0, 255, 0)
gris = (200, 200, 200)
azul_oscuro = (0, 0, 139)
rojo_oscuro = (139, 0, 0)
verde_oscuro = (0, 100, 0)
amarillo = (255, 255, 0)
naranja = (255, 165, 0)
morado = (128, 0, 128)
cyan = (0, 255, 255)
magenta = (255, 0, 255)
turquesa = (64, 224, 208)
verde_claro = (144, 238, 144)
rosa = (255, 192, 203)

#..... fuentes
fuente = pygame.font.SysFont('Arial', 16)
fuente_grande = pygame.font.SysFont('Arial', 24)
fuente_muy_grande = pygame.font.SysFont('Arial', 28)

#..... parámetros de simulación
N = 200  #..... resolución de la cuadrícula
diff_init = 0.0001  #..... difusividad inicial
visc_init = 0.0001  #..... viscosidad inicial
dt_init = 0.1  #..... intervalo de tiempo inicial

#..... tamaño del sub-canvas
SUB_CANVAS_SIZE = 600  #..... sub-canvas de 600x600
cell_width = SUB_CANVAS_SIZE // N
cell_height = SUB_CANVAS_SIZE // N

#..... posiciones de los elementos de la interfaz
sub_canvas_rect = pygame.Rect(50, 50, SUB_CANVAS_SIZE, SUB_CANVAS_SIZE)  #..... área para la simulación
gui_rect = pygame.Rect(750, 30, 510, 660)  #..... área para controles y visualizaciones

#..... configuración de fps
fps = 60
clock = pygame.time.Clock()

#.........................................................| 2. Clases y funciones auxiliares
class Fluid:
    """
    Clase para la simulación de fluidos basada en las ecuaciones de navier-stokes
    
    ----------
    El fluido se modela como un campo de velocidad (Vx, Vy) y densidad en una cuadrícula de tamaño N x N.
    Las ecuaciones de Navier-Stokes se resuelven mediante el método de proyección de presión.
    ----------
    Por otro lado, la difusión y advección de las propiedades del fluido se realizan de manera explícita.
    ----------
    """

    def __init__(self, N, diffusion, viscosity, dt):
        self.N = N
        self.diff = diffusion
        self.visc = viscosity
        self.dt = dt

        self.size = N + 2  #..... incluye bordes
        self.s = np.zeros((self.size, self.size))
        self.density = np.zeros((self.size, self.size))

        self.Vx = np.zeros((self.size, self.size))
        self.Vy = np.zeros((self.size, self.size))

        self.Vx0 = np.zeros((self.size, self.size))
        self.Vy0 = np.zeros((self.size, self.size))

    def add_density(self, x, y, amount):
        """agrega densidad en la posición (x, y)"""
        if 1 <= x < self.size - 1 and 1 <= y < self.size - 1:
            self.density[y, x] += amount

    def add_velocity(self, x, y, amount_x, amount_y):
        """agrega velocidad en la posición (x, y)"""
        if 1 <= x < self.size - 1 and 1 <= y < self.size - 1:
            self.Vx[y, x] += amount_x
            self.Vy[y, x] += amount_y

    def step(self):
        """avanza la simulación un paso temporal"""
        self.diffuse(1, self.Vx0, self.Vx, self.visc, self.dt)
        self.diffuse(2, self.Vy0, self.Vy, self.visc, self.dt)

        self.project(self.Vx0, self.Vy0, self.Vx, self.Vy)

        self.advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt)
        self.advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, self.dt)

        self.project(self.Vx, self.Vy, self.Vx0, self.Vy0)

        self.diffuse(0, self.s, self.density, self.diff, self.dt)
        self.advect(0, self.density, self.s, self.Vx, self.Vy, self.dt)

    def diffuse(self, b, x, x0, diff, dt):
        """dispersion de las propiedades del fluido"""
        a = dt * diff * self.N * self.N
        self.lin_solve(b, x, x0, a, 1 + 4 * a)

    def lin_solve(self, b, x, x0, a, c):
        """solución lineal por iteración"""
        for _ in range(20):
            x[1:-1,1:-1] = (x0[1:-1,1:-1] + a*(x[1:-1,0:-2] + x[1:-1,2:] +
                                               x[0:-2,1:-1] + x[2:,1:-1])) / c
            self.set_bnd(b, x)

    def project(self, velocX, velocY, p, div):
        """proyección para mantener la divergencia cero"""
        div[1:-1,1:-1] = -0.5*(velocX[1:-1,2:] - velocX[1:-1,0:-2] +
                                velocY[2:,1:-1] - velocY[0:-2,1:-1])/self.N
        p.fill(0)
        self.lin_solve(0, p, div, 1, 4)

        velocX[1:-1,1:-1] -= 0.5*(p[1:-1,2:] - p[1:-1,0:-2]) * self.N
        velocY[1:-1,1:-1] -= 0.5*(p[2:,1:-1] - p[0:-2,1:-1]) * self.N
        self.set_bnd(1, velocX)
        self.set_bnd(2, velocY)

    def advect(self, b, d, d0, velocX, velocY, dt):
        """transporte de las propiedades del fluido"""
        N = self.N
        dt0 = dt * N
        X, Y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        x = X - dt0 * velocX
        y = Y - dt0 * velocY

        x = np.clip(x, 0.5, self.size - 1.5)
        y = np.clip(y, 0.5, self.size - 1.5)

        i0 = np.floor(x).astype(int)
        i1 = i0 + 1
        j0 = np.floor(y).astype(int)
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1 - s1
        t1 = y - j0
        t0 = 1 - t1

        i0 = np.clip(i0, 0, self.size - 1)
        i1 = np.clip(i1, 0, self.size - 1)
        j0 = np.clip(j0, 0, self.size - 1)
        j1 = np.clip(j1, 0, self.size - 1)

        d[1:-1,1:-1] = (s0[1:-1,1:-1]*(t0[1:-1,1:-1]*d0[j0[1:-1,1:-1],i0[1:-1,1:-1]] +
                                        t1[1:-1,1:-1]*d0[j1[1:-1,1:-1],i0[1:-1,1:-1]]) +
                           s1[1:-1,1:-1]*(t0[1:-1,1:-1]*d0[j0[1:-1,1:-1],i1[1:-1,1:-1]] +
                                        t1[1:-1,1:-1]*d0[j1[1:-1,1:-1],i1[1:-1,1:-1]]))
        self.set_bnd(b, d)

    def set_bnd(self, b, x):
        """establece las condiciones de frontera"""
        N = self.N
        #..... bordes
        x[0,1:-1] = -x[1,1:-1] if b == 2 else x[1,1:-1]
        x[-1,1:-1] = -x[-2,1:-1] if b == 2 else x[-2,1:-1]
        x[1:-1,0] = -x[1:-1,1] if b == 1 else x[1:-1,1]
        x[1:-1,-1] = -x[1:-1,-2] if b == 1 else x[1:-1,-2]

        #..... esquinas
        x[0,0] = 0.5 * (x[1,0] + x[0,1])
        x[0,-1] = 0.5 * (x[1,-1] + x[0,-2])
        x[-1,0] = 0.5 * (x[-2,0] + x[-1,1])
        x[-1,-1] = 0.5 * (x[-2,-1] + x[-1,-2])

#.........................................................| 3. GUI

class Slider:
    """clase para los sliders de la interfaz"""

    def __init__(self, x, y, w, h, min_val, max_val, start, label, step=0.0001):
        self.rect = pygame.Rect(x, y, w, h)
        self.min = min_val
        self.max = max_val
        self.value = start
        self.label = label
        self.step = step
        self.handle_width = 16
        self.handle_rect = pygame.Rect(x, y, self.handle_width, h)
        self.actualizaeventos()
        self.dragging = False

    def actualizaeventos(self):
        """actualiza la posición del manejador del slider"""
        ratio = (self.value - self.min) / (self.max - self.min)
        self.handle_rect.x = self.rect.x + int(ratio * (self.rect.width - self.handle_width))

    def draw(self, surface):
        """dibuja el slider en la superficie"""
        #..... dibujar la barra
        pygame.draw.rect(surface, gris, self.rect)
        #..... dibujar el manejador
        color_manejador = rojo_oscuro if self.dragging else rojo
        pygame.draw.rect(surface, color_manejador, self.handle_rect, border_radius=8)
        #..... dibujar la etiqueta
        etiqueta = fuente.render(f"{self.label}: {self.value:.4f}", True, blanco)
        surface.blit(etiqueta, (self.rect.x, self.rect.y - 25))
        #..... dibujar los ticks
        for i in range(11):
            tick_x = self.rect.x + i * (self.rect.width // 10)
            pygame.draw.line(surface, blanco, (tick_x, self.rect.y + self.rect.height),
                             (tick_x, self.rect.y + self.rect.height + 10), 2)

    def sobreeventos(self, event):
        """maneja los eventos del slider"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                new_x = max(self.rect.x, min(event.pos[0], self.rect.x + self.rect.width - self.handle_width))
                self.handle_rect.x = new_x
                ratio = (self.handle_rect.x - self.rect.x) / (self.rect.width - self.handle_width)
                new_value = self.min + ratio * (self.max - self.min)
                #..... ajustar al step
                self.value = round(new_value / self.step) * self.step
                self.value = max(self.min, min(self.value, self.max))
                self.actualizaeventos()

class Button:
    """clase para los botones de la interfaz"""

    def __init__(self, x, y, w, h, text, color=azul, hover_color=azul_oscuro, text_color=blanco, icon_path=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = fuente_grande
        self.hovered = False
        self.icon = None
        if icon_path and os.path.exists(icon_path):
            self.icon = pygame.image.load(icon_path)
            self.icon = pygame.transform.scale(self.icon, (h - 10, h - 10))  #..... ajustar tamaño

    def draw(self, surface):
        """dibuja el botón en la superficie"""
        current_color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, current_color, self.rect, border_radius=8)
        #..... dibujar el texto
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        #..... desplazar el texto si hay ícono
        if self.icon:
            text_rect.x += self.icon.get_width() // 2 + 5
            surface.blit(self.icon, (self.rect.x + 5, self.rect.y + 5))
        surface.blit(text_surf, text_rect)

    def sobreeventos(self, event):
        """maneja los eventos del botón"""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                return True
        return False

#.........................................................| 4. Algoritmos e inicialización de la simulación
def trazadensity(surface, fluid, rect, mode='vector'):
    """
    Dibuja la densidad del fluido según el modo de visualización.
    
    Parámetros:
    -----------
    surface : pygame.Surface
        Superficie de pygame donde se dibujará la densidad del fluido.
    fluid : Fluid
        Instancia de la clase Fluid que contiene el estado actual de la simulación.
    rect : pygame.Rect
        Rectángulo que define el área de la simulación dentro de la ventana de pygame.
    mode : str
        Modo de visualización de la densidad ('vector', 'heatmap', 'gradient', 'combined', 'streamlines', 'vorticity').
    """
    density = fluid.density
    N = fluid.N
    if mode == 'heatmap':
        #..... normalizar densidad para mapeo de colores
        max_d = np.max(density)
        if max_d == 0:
            max_d = 1
        normalized = density / max_d
        #..... crear superficie de densidad
        density_surface = pygame.Surface((rect.width, rect.height))
        for y in range(1, N + 1):
            for x in range(1, N + 1):
                d = normalized[y, x]
                if d > 0:
                    #..... mapa de colores basado en densidad (de azul a rojo)
                    color = pygame.Color(0, 0, 255)
                    color.hsva = (240 - (240 * d), 100, 100, 100)
                    pygame.draw.rect(density_surface, color,
                                     ( (x -1)*cell_width, (y -1)*cell_height, cell_width, cell_height))
        #..... aplicar suavizado
        density_surface = pygame.transform.smoothscale(density_surface, (rect.width, rect.height))
        surface.blit(density_surface, rect)
    elif mode == 'gradient':
        #..... gradiente de colores más suave
        max_d = np.max(density)
        if max_d == 0:
            max_d = 1
        normalized = density / max_d
        density_surface = pygame.Surface((rect.width, rect.height))
        for y in range(1, N + 1):
            for x in range(1, N + 1):
                d = normalized[y, x]
                if d > 0:
                    r = min(int(255 * d), 255)
                    g = min(int(255 * (1 - d)), 255)
                    b = 128
                    color = (r, g, b)
                    pygame.draw.rect(density_surface, color,
                                     ( (x -1)*cell_width, (y -1)*cell_height, cell_width, cell_height))
        density_surface = pygame.transform.smoothscale(density_surface, (rect.width, rect.height))
        surface.blit(density_surface, rect)
    elif mode == 'combined':
        #..... combinar heatmap y vectores
        max_d = np.max(density)
        if max_d == 0:
            max_d = 1
        normalized = density / max_d
        density_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for y in range(1, N + 1):
            for x in range(1, N + 1):
                d = normalized[y, x]
                if d > 0:
                    color = pygame.Color(0, 0, 255)
                    color.hsva = (240 - (240 * d), 100, 100, 100)
                    pygame.draw.rect(density_surface, color,
                                     ( (x -1)*cell_width, (y -1)*cell_height, cell_width, cell_height))
        #..... aplicar transparencia para combinar con vectores
        density_surface.set_alpha(150)
        surface.blit(density_surface, rect)
    elif mode == 'streamlines':
        #..... visualización de líneas de corriente
        Vx = fluid.Vx
        Vy = fluid.Vy
        for y in range(1, N + 1, 10):
            for x in range(1, N + 1, 10):
                start_pos = (rect.x + (x -1)*cell_width + cell_width//2, rect.y + (y -1)*cell_height + cell_height//2)
                path = []
                pos = np.array([x, y], dtype=float)
                for _ in range(20):
                    vx = Vx[int(pos[1]), int(pos[0])]
                    vy = Vy[int(pos[1]), int(pos[0])]
                    norm = math.sqrt(vx**2 + vy**2)
                    if norm == 0:
                        break
                    pos += (vx / norm, vy / norm) * 0.5  #..... paso pequeño
                    if pos[0] < 1 or pos[0] > N or pos[1] < 1 or pos[1] > N:
                        break
                    path.append((rect.x + (pos[0]-1)*cell_width + cell_width//2,
                                 rect.y + (pos[1]-1)*cell_height + cell_height//2))
                if len(path) > 1:
                    pygame.draw.lines(surface, morado, False, path, 1)
    elif mode == 'vorticity':
        #..... visualización de la vorticidad (rotacional del campo de velocidad)
        Vx = fluid.Vx
        Vy = fluid.Vy
        vort = (Vx[2:,1:-1] - Vx[:-2,1:-1]) - (Vy[1:-1,2:] - Vy[1:-1,:-2])
        max_vort = np.max(np.abs(vort))
        if max_vort == 0:
            max_vort = 1
        normalized = vort / max_vort
        vort_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        for y in range(1, N +1):
            for x in range(1, N +1):
                v = normalized[y-1, x-1]
                if v != 0:
                    #..... mapea vorticidad a colores cian y magenta
                    if v > 0:
                        color = pygame.Color(0, 255, 255)
                        color.hsva = (180, 100, 100, 100)
                    else:
                        color = pygame.Color(255, 0, 255)
                        color.hsva = (300, 100, 100, 100)
                    pygame.draw.rect(vort_surface, color,
                                     ( (x-1)*cell_width, (y-1)*cell_height, cell_width, cell_height))
        vort_surface.set_alpha(120)
        surface.blit(vort_surface, rect)

def trazavelocity(surface, fluid, rect):
    """
    Dibuja los vectores de velocidad del fluido.
    
    Parámetros:
    -----------
    surface : pygame.Surface
        Superficie de pygame donde se dibujarán los vectores de velocidad.
    fluid : Fluid
        Instancia de la clase Fluid que contiene el estado actual de la simulación.
    rect : pygame.Rect
        Rectángulo que define el área de la simulación dentro de la ventana de pygame.    
    """
    Vx = fluid.Vx
    Vy = fluid.Vy
    N = fluid.N
    skip = max(N // 50, 1)  #..... ajustar para evitar demasiados vectores
    for y in range(1, N + 1, skip):
        for x in range(1, N + 1, skip):
            vx = Vx[y, x]
            vy = Vy[y, x]
            speed = math.sqrt(vx ** 2 + vy ** 2)
            if speed > 0:
                #..... escalar la longitud de las flechas
                scale = 10
                start_pos = (rect.x + (x - 1) * cell_width + cell_width // 2,
                             rect.y + (y - 1) * cell_height + cell_height // 2)
                end_pos = (start_pos[0] + int(vx * scale), start_pos[1] + int(vy * scale))
                pygame.draw.line(surface, verde_claro, start_pos, end_pos, 2)
                #..... dibujar una flecha en el extremo
                angle = math.atan2(vy, vx)
                arrow_size = 6
                arrow_angle = math.pi / 6
                left = (end_pos[0] - arrow_size * math.cos(angle - arrow_angle),
                        end_pos[1] - arrow_size * math.sin(angle - arrow_angle))
                right = (end_pos[0] - arrow_size * math.cos(angle + arrow_angle),
                         end_pos[1] - arrow_size * math.sin(angle + arrow_angle))
                pygame.draw.polygon(surface, verde_claro, [end_pos, left, right])

def reseteavsual(fluid):
    """reinicia la simulación limpiando todas las matrices"""
    fluid.density.fill(0)
    fluid.Vx.fill(0)
    fluid.Vy.fill(0)
    fluid.Vx0.fill(0)
    fluid.Vy0.fill(0)
    fluid.s.fill(0)

def colocaayuda(surface, rect):
    """dibuja la sección de ayuda con las instrucciones"""
    textos = [
        "Simulador de fluidos basado en las ecuaciones de navier-stokes.",
        "Ayuda:",
        "1. Ajusta los parámetros de difusividad, viscosidad y Δt utilizando los sliders.",
        "2. Presiona 'iniciar' para comenzar la simulación.",
        "3. Presiona 'pausar' para detener/reanudar la simulación.",
        "4. Presiona 'reiniciar' para limpiar la simulación.",
        "5. Haz clic y arrastra en el sub-canvas para agregar densidad y velocidad.",
        "6. Cambia el modo de visualización utilizando los botones correspondientes.",
        "7. Observa las variables en tiempo real en la sección de variables.",
    ]
    y_offset = rect.y + 20
    for texto in textos:
        render = fuente.render(texto, True, blanco)
        surface.blit(render, (rect.x + 20, y_offset))
        y_offset += 20

def dibujavariable(surface, fluid, rect):
    """dibuja las variables de la simulación en tiempo real"""
    textos = [
        f"Difusividad: {fluid.diff:.5f}",
        f"Viscosidad: {fluid.visc:.5f}",
        f"Δt: {fluid.dt:.2f}",
        f"Densidad total: {np.sum(fluid.density):.2f}",
        f"Velocidad promedio x: {np.mean(fluid.Vx):.5f}",
        f"Velocidad promedio y: {np.mean(fluid.Vy):.5f}",
    ]
    y_offset = rect.y + 10
    for texto in textos:
        render = fuente.render(texto, True, blanco)
        surface.blit(render, (rect.x + 20, y_offset))
        y_offset += 25

def dibujacoordenadas(surface, rect):
    """dibuja los ejes coordenados en el sub-canvas"""
    #..... dibujar ejes x e y
    pygame.draw.line(surface, amarillo, (rect.x, rect.centery), (rect.x + rect.width, rect.centery), 2)
    pygame.draw.line(surface, amarillo, (rect.centerx, rect.y), (rect.centerx, rect.y + rect.height), 2)

    #..... dibujar etiquetas
    label_x = fuente.render("x", True, amarillo)
    label_y = fuente.render("y", True, amarillo)
    surface.blit(label_x, (rect.x + rect.width - 30, rect.centery + 10))
    surface.blit(label_y, (rect.centerx + 10, rect.y + 10))

def dibujapresion(surface, fluid, rect):
    """
    Dibuja el campo de presión del fluido de manera optimizada.

    Parámetros:
    -----------
    surface : pygame.Surface
        Superficie de pygame donde se dibujará el campo de presión.
    fluid : Fluid
        Instancia de la clase Fluid que contiene el estado actual de la simulación.
    rect : pygame.Rect
        Rectángulo que define el área de la simulación dentro de la ventana de pygame.
    """
    pressure = fluid.s
    N = fluid.N
    max_p = np.max(np.abs(pressure))
    if max_p == 0:
        max_p = 1
    normalized = pressure / max_p
    pressure_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    for y in range(1, N + 1):
        for x in range(1, N + 1):
            p = normalized[y, x]
            if p != 0:
                #..... mapea presión a colores (verde para positiva, rojo para negativa)
                if p > 0:
                    color = pygame.Color(0, 255, 0)
                    color.hsva = (120, 100, 100, 100)
                else:
                    color = pygame.Color(255, 0, 0)
                    color.hsva = (0, 100, 100, 100)
                pygame.draw.rect(pressure_surface, color,
                                 ( (x -1)*cell_width, (y -1)*cell_height, cell_width, cell_height))
    pressure_surface.set_alpha(150)
    surface.blit(pressure_surface, rect)
    
def dibujastreamlines(surface, fluid, rect):
    """
    Dibuja líneas de corriente del flujo de manera optimizada dentro de lo posible.

    Parámetros:
    -----------
    surface : pygame.Surface
        Superficie de pygame donde se dibujarán las líneas de corriente.
    fluid : Fluid
        Instancia de la clase Fluid que contiene el estado actual de la simulación.
    rect : pygame.Rect
        Rectángulo que define el área de la simulación dentro de la ventana de pygame.
    """
    Vx = fluid.Vx
    Vy = fluid.Vy
    N = fluid.N
    step = 5  #..... paso entre puntos semilla

    cell_width = rect.width / N
    cell_height = rect.height / N

    rect_x = rect.x
    rect_y = rect.y

    #..... define el color de las líneas de corriente
    turquesa = (64, 224, 208)

    #..... con esto convertimoss las matrices de velocities a listas de listas para acceso más rápido en Python puro
    Vx_list = Vx.tolist()
    Vy_list = Vy.tolist()

    #..... reutiliza una lista para las trayectorias
    path = []

    for y in range(1, N + 1, step):
        for x in range(1, N + 1, step):
            path.clear()  #..... limpia la lista para la nueva trayectoria
            pos_x = float(x)
            pos_y = float(y)

            for _ in range(30):
                ix = int(pos_x)
                iy = int(pos_y)

                #..... tanteamos si la posición está dentro de los límites
                if ix < 1 or ix > N or iy < 1 or iy > N:
                    break

                #..... entonces obtenemos las componentes de velocidad en la posición actual
                vx = Vx_list[iy][ix]
                vy = Vy_list[iy][ix]

                #..... calcula la norma (magnitud) de la velocidad
                norm_sq = vx * vx + vy * vy
                if norm_sq == 0:
                    break

                #..... calcula el inverso de la norma para evitar llamar a sqrt y dividir
                inv_norm = 1.0 / math.sqrt(norm_sq)

                #..... ahora hacemos el calculo del paso pequeño en la dirección de la velocidad normalizada
                delta_x = vx * inv_norm * 0.4
                delta_y = vy * inv_norm * 0.4

                pos_x += delta_x
                pos_y += delta_y

                #..... se chequea nuevamente los límites después del paso
                if pos_x < 1 or pos_x > N or pos_y < 1 or pos_y > N:
                    break

                #..... convierte las coordenadas de la cuadrícula a píxeles
                pixel_x = rect_x + int((pos_x - 1.0) * cell_width + cell_width * 0.5)
                pixel_y = rect_y + int((pos_y - 1.0) * cell_height + cell_height * 0.5)

                path.append((pixel_x, pixel_y))

            #..... dibuja la línea de corriente si tiene más de un punto
            if len(path) > 1:
                pygame.draw.lines(surface, turquesa, False, path, 1)
                
def main():
    """función principal que ejecuta el simulador"""

    #..... instancia de la simulación
    fluid = Fluid(N, diff_init, visc_init, dt_init)

    #..... instancias de sliders
    slider_diff = Slider(770, 70, 200, 20, 0.000, 0.001, diff_init, "Difusividad", step=0.0001)
    slider_visc = Slider(770, 140, 200, 20, 0.000, 0.001, visc_init, "Viscosidad", step=0.0001)
    slider_dt = Slider(770, 210, 200, 20, 0.01, 1.0, dt_init, "Δt", step=0.01)

    #..... instancias de botones
    button_start = Button(770, 290, 100, 40, "Iniciar")
    button_pause = Button(890, 290, 100, 40, "Pausar")
    button_reset = Button(1010, 290, 100, 40, "Reiniciar")
    button_help = Button(1130, 290, 100, 40, "Ayuda")
    button_visual_vector = Button(770, 380, 100, 40, "Vectores")
    button_visual_heatmap = Button(890, 380, 100, 40, "Heatmap")
    button_visual_gradient = Button(1010, 380, 100, 40, "Gradiente")
    button_visual_combined = Button(770, 440, 100, 40, "Mixto")
    button_visual_streamlines = Button(890, 440, 100, 40, "Streamlines")
    button_visual_vorticity = Button(1010, 440, 100, 40, "Vorticity")
    button_visual_pressure = Button(1130, 380, 100, 40, "Presión")

    #..... estado de la simulación
    simulando = False
    pausado = False
    mostrar_ayuda = False
    modo_visualizacion = 'vector'  #..... opciones: 'vector', 'heatmap', 'gradient', 'combined', 'streamlines', 'vorticity', 'pressure'

    #..... loop principal
    running = True
    while running:
        clock.tick(fps)  #..... limitar a fps
        WINDOW.fill(negro)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            #..... manejar eventos de sliders
            slider_diff.sobreeventos(event)
            slider_visc.sobreeventos(event)
            slider_dt.sobreeventos(event)

            #..... manejar eventos de botones
            if button_start.sobreeventos(event):
                simulando = True
                pausado = False
            if button_pause.sobreeventos(event):
                if simulando:
                    pausado = not pausado
            if button_reset.sobreeventos(event):
                reseteavsual(fluid)
                simulando = False
                pausado = False
            if button_help.sobreeventos(event):
                mostrar_ayuda = not mostrar_ayuda
            if button_visual_vector.sobreeventos(event):
                modo_visualizacion = 'vector'
            if button_visual_heatmap.sobreeventos(event):
                modo_visualizacion = 'heatmap'
            if button_visual_gradient.sobreeventos(event):
                modo_visualizacion = 'gradient'
            if button_visual_combined.sobreeventos(event):
                modo_visualizacion = 'combined'
            if button_visual_streamlines.sobreeventos(event):
                modo_visualizacion = 'streamlines'
            if button_visual_vorticity.sobreeventos(event):
                modo_visualizacion = 'vorticity'
            if button_visual_pressure.sobreeventos(event):
                modo_visualizacion = 'pressure'

        #..... actualizar los parámetros de la simulación desde los sliders
        fluid.diff = slider_diff.value
        fluid.visc = slider_visc.value
        fluid.dt = slider_dt.value

        #..... interacción con el mouse en el sub-canvas
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if sub_canvas_rect.collidepoint(mx, my):
                grid_x = (mx - sub_canvas_rect.x) // cell_width + 1
                grid_y = (my - sub_canvas_rect.y) // cell_height + 1
                fluid.add_density(grid_x, grid_y, 100)
                #..... agregar un pequeño impulso de velocidad basado en el movimiento del mouse
                fluid.add_velocity(grid_x, grid_y, (mx - sub_canvas_rect.centerx) * 0.05,
                                   (my - sub_canvas_rect.centery) * 0.05)

        #..... avanzar la simulación si está en estado de simulación y no está pausada
        if simulando and not pausado:
            fluid.step()

        #..... dibujar el sub-canvas con bordes
        pygame.draw.rect(WINDOW, gris, sub_canvas_rect, 2)

        #..... dibujar visualizaciones según el modo seleccionado
        if modo_visualizacion == 'vector':
            trazavelocity(WINDOW, fluid, sub_canvas_rect)
        elif modo_visualizacion == 'heatmap':
            trazadensity(WINDOW, fluid, sub_canvas_rect, mode='heatmap')
        elif modo_visualizacion == 'gradient':
            trazadensity(WINDOW, fluid, sub_canvas_rect, mode='gradient')
        elif modo_visualizacion == 'combined':
            trazadensity(WINDOW, fluid, sub_canvas_rect, mode='combined')
            trazavelocity(WINDOW, fluid, sub_canvas_rect)
        elif modo_visualizacion == 'streamlines':
            dibujastreamlines(WINDOW, fluid, sub_canvas_rect)
        elif modo_visualizacion == 'vorticity':
            trazadensity(WINDOW, fluid, sub_canvas_rect, mode='vorticity')
        elif modo_visualizacion == 'pressure':
            dibujapresion(WINDOW, fluid, sub_canvas_rect)

        #..... dibujar el gui
        pygame.draw.rect(WINDOW, (50, 50, 50), gui_rect, border_radius=15)
        #..... dibujar sliders
        slider_diff.draw(WINDOW)
        slider_visc.draw(WINDOW)
        slider_dt.draw(WINDOW)
        #..... dibujar botones
        button_start.draw(WINDOW)
        button_pause.draw(WINDOW)
        button_reset.draw(WINDOW)
        button_help.draw(WINDOW)
        button_visual_vector.draw(WINDOW)
        button_visual_heatmap.draw(WINDOW)
        button_visual_gradient.draw(WINDOW)
        button_visual_combined.draw(WINDOW)
        button_visual_streamlines.draw(WINDOW)
        button_visual_vorticity.draw(WINDOW)
        button_visual_pressure.draw(WINDOW)
        #..... dibujar variables
        variables_rect = pygame.Rect(770, 510, 400, 165)
        pygame.draw.rect(WINDOW, (70, 70, 70), variables_rect, border_radius=10)
        dibujavariable(WINDOW, fluid, variables_rect)
        #..... dibujar ayuda si está activada
        if mostrar_ayuda:
            ayuda_rect = pygame.Rect(770, 370, 480, 220)
            pygame.draw.rect(WINDOW, (30, 30, 30), ayuda_rect, border_radius=10)
            colocaayuda(WINDOW, ayuda_rect)
        #..... dibujar coordenadas en el sub-canvas
        dibujacoordenadas(WINDOW, sub_canvas_rect)
        #..... mostrar fps
        fps_text = fuente.render(f"fps: {int(clock.get_fps())}", True, blanco)
        WINDOW.blit(fps_text, (WIDTH - 80, HEIGHT - 60))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

#.........................................................| FIN DEL CÓDIGO
# Derechos de autor (c) FECORO 2022.
