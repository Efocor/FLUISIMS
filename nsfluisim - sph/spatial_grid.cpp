/*
Este es el código de un grid, o sea un ejemplo de una cuadrícula espacial que se usa en simulaciones 
de fluidos para acelerar la búsqueda de vecinos. La cuadrícula divide el espacio en celdas de tamaño fijo 
y asigna cada partícula a una celda. Luego, para encontrar los vecinos de una partícula, 
solo se buscan en las celdas adyacentes en lugar de comparar con todas las partículas. 
Este enfoque reduce la complejidad de tiempo de O(n^2) a O(n) en el cálculo de fuerzas entre partículas cercanas. 


El código muestra la implementación de la cuadrícula y cómo se actualiza y se buscan los vecinos de una partícula. 
Este código es parte de una simulación de fluidos utilizando el método Smoothed Particle Hydrodynamics (SPH) en C++.
*/

//... spatial_grid.cpp
#include "spatial_grid.h"

SpatialGrid::SpatialGrid(int width, int height, float cellSize) 
    : gridWidth(width/cellSize), gridHeight(height/cellSize), cellSize(cellSize) {
    grid.resize(gridWidth * gridHeight);
}

void SpatialGrid::updateGrid(const std::vector<Particle>& particles) {
    //... limpiar grid
    for (auto& cell : grid) {
        cell.particleIndices.clear();
    }
    
    //... insertar partículas en el grid
    for (int i = 0; i < particles.size(); i++) {
        int cellX = particles[i].position.x / cellSize;
        int cellY = particles[i].position.y / cellSize;
        
        if (cellX >= 0 && cellX < gridWidth && cellY >= 0 && cellY < gridHeight) {
            grid[cellY * gridWidth + cellX].particleIndices.push_back(i);
        }
    }
}

std::vector<int> SpatialGrid::getNeighbors(const sf::Vector2f& position) {
    std::vector<int> neighbors;
    int cellX = position.x / cellSize;
    int cellY = position.y / cellSize;
    
    //... buscar en celdas adyacentes
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cellX + dx;
            int ny = cellY + dy;
            
            if (nx >= 0 && nx < gridWidth && ny >= 0 && ny < gridHeight) {
                const auto& cell = grid[ny * gridWidth + nx];
                neighbors.insert(neighbors.end(), 
                               cell.particleIndices.begin(), 
                               cell.particleIndices.end());
            }
        }
    }
    return neighbors;
}
