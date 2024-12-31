/*
Este archivo contiene la implementación de la clase SPHSolver. 

Funciona tal que se calculan las fuerzas de presión y viscosidad que actúan sobre cada partícula.
Se integran las posiciones y velocidades de las partículas en función de las fuerzas calculadas.

Este código es solo una implementación básica y no optimizada de una simulación de fluidos.
*/

//... sph_solver.cpp
#include "sph_solver.h"
#include <cmath>

float SPHSolver::kernelPoly6(float r, float h) {
    if (r > h) return 0.0f;
    float term = h * h - r * r;
    return 315.0f / (64.0f * M_PI * pow(h, 9)) * term * term * term;
}

float SPHSolver::kernelSpikyGradient(float r, float h) {
    if (r > h) return 0.0f;
    float term = h - r;
    return -45.0f / (M_PI * pow(h, 6)) * term * term;
}

float SPHSolver::kernelViscosityLaplacian(float r, float h) {
    if (r > h) return 0.0f;
    return 45.0f / (M_PI * pow(h, 6)) * (h - r);
}

void SPHSolver::calculateDensityPressure(ParticleSystem& particleSystem) {
    auto& particles = particleSystem.getParticles();
    float h = particleSystem.getSmoothingLength();
    float mass = particleSystem.getParticleMass();
    
    for (auto& particle : particles) {
        particle.density = 0.0f;
        for (const auto& other : particles) {
            sf::Vector2f diff = particle.position - other.position;
            float r = sqrt(diff.x * diff.x + diff.y * diff.y);
            particle.density += mass * kernelPoly6(r, h);
        }
        particle.pressure = stiffness * (particle.density - restDensity);
    }
}

void SPHSolver::calculateForces(ParticleSystem& particleSystem) {
    auto& particles = particleSystem.getParticles();
    float h = particleSystem.getSmoothingLength();
    float mass = particleSystem.getParticleMass();
    
    for (auto& particle : particles) {
        sf::Vector2f pressureForce(0.0f, 0.0f);
        sf::Vector2f viscosityForce(0.0f, 0.0f);
        
        for (const auto& other : particles) {
            if (&other == &particle) continue;
            
            sf::Vector2f diff = particle.position - other.position;
            float r = sqrt(diff.x * diff.x + diff.y * diff.y);
            if (r < h) {
                //... fuerza de presión
                float pressureGrad = kernelSpikyGradient(r, h);
                pressureForce += diff/r * mass * 
                    (particle.pressure + other.pressure)/(2.0f * other.density) * 
                    pressureGrad;
                
                //... fuerza de viscosidad
                float viscLap = kernelViscosityLaplacian(r, h);
                viscosityForce += mass * (other.velocity - particle.velocity) / 
                    other.density * viscLap;
            }
        }
        
        sf::Vector2f gravity(0.0f, 981.0f); //... gravedad en cm/s^2
        particle.force = pressureForce * -1.0f + 
                        viscosityForce * viscosity + 
                        gravity;
    }
}

void SPHSolver::update(ParticleSystem& particleSystem) {
    calculateDensityPressure(particleSystem);
    calculateForces(particleSystem);
}
