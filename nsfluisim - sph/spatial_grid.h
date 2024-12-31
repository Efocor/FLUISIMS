// spatial_grid.h
#pragma once
#include <vector>
#include "particle_system.h"

class SpatialGrid {
private:
    struct Cell {
        std::vector<int> particleIndices;
    };
    
    std::vector<Cell> grid;
    int gridWidth, gridHeight;
    float cellSize;

public:
    SpatialGrid(int width, int height, float cellSize);
    void updateGrid(const std::vector<Particle>& particles);
    std::vector<int> getNeighbors(const sf::Vector2f& position);
};
