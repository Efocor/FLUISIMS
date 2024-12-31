// sph_solver.h
#pragma once
#include "particle_system.h"
#include "spatial_grid.h"

class SPHSolver {
private:
    float viscosity;
    float stiffness;
    float restDensity;
    float deltaTime;
    SpatialGrid grid;
    
    float kernelPoly6(float r, float h);
    float kernelSpikyGradient(float r, float h);
    float kernelViscosityLaplacian(float r, float h);

public:
    SPHSolver() 
        : viscosity(250.0f),
          stiffness(50.0f),
          restDensity(1000.0f),
          deltaTime(1.0f/60.0f),
          grid(WINDOW_WIDTH, WINDOW_HEIGHT, 30.0f) {}

    void update(ParticleSystem& particles);
    void calculateDensityPressure(ParticleSystem& particles);
    void calculateForces(ParticleSystem& particles);
};
