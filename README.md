# FLUISIMS
Este proyecto implementa una simulación de fluidos basada en las **Ecuaciones de Navier-Stokes** utilizando **Python** y **Pygame** para la visualización interactiva. La simulación modela el fluido como un campo de velocidad `(Vx, Vy)` y densidad en una cuadrícula de tamaño `N x N`, resolviendo las ecuaciones mediante el método de proyección de presión. Además, la difusión y advección de las propiedades del fluido se realizan de manera explícita para garantizar una representación precisa del comportamiento del fluido.

## Características Principales

- **Visualización Interactiva**: Utiliza **Pygame** para dibujar y actualizar en tiempo real la simulación del fluido, proporcionando una experiencia visual intuitiva.
  
- **Interfaz de Usuario**: Incluye botones interactivos que permiten ajustar parámetros clave como la difusividad y la viscosidad del fluido, facilitando la experimentación y comprensión de los efectos de estos parámetros en la simulación.
  
- **Clases y Funciones Auxiliares**: La estructura del código está organizada en clases como `Fluid` que encapsulan la lógica de la simulación, así como funciones auxiliares para dibujar coordenadas y gestionar la interfaz gráfica.
  
- **Optimización Computacional**: Emplea **NumPy** para operaciones matriciales eficientes, mejorando el rendimiento de la simulación sobre la cuadrícula.

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal para implementar la lógica de la simulación.
- **Pygame**: Biblioteca utilizada para la visualización y creación de la interfaz gráfica.
- **MatPlotLib**: Biblioteca que permite todo tipo de visualizaciones en gráfico y otros.
- **NumPy**: Librería fundamental para operaciones numéricas y manejo eficiente de matrices.
