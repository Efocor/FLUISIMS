//... particle_system.cpp

/*
Aquí se implementa la lógica de la simulación de partículas y obstáculos.

Esta funciona tal que se inicializan las partículas en una cuadrícula y se actualizan
sus posiciones y velocidades en cada frame. Se manejan colisiones con los bordes de la ventana
y con obstáculos estáticos.

Además, se calcula la velocidad promedio de las partículas y se guarda un historial de
velocidades para su visualización en una gráfica.
*/
#include "particle_system.h"
#include <random>

void ParticleSystem::initializeParticles(int startX, int startY) {
    const int particlesPerRow = 30;
    const int particlesPerCol = 30;
    const float spacing = 8.0f;
    
    for (int y = 0; y < particlesPerCol; y++) {
        for (int x = 0; x < particlesPerRow; x++) {
            Particle p;
            p.position = sf::Vector2f(startX + x * spacing, startY + y * spacing);
            p.velocity = sf::Vector2f(0.0f, 0.0f);
            p.force = sf::Vector2f(0.0f, 0.0f);
            p.density = 0.0f;
            p.pressure = 0.0f;
            p.color = sf::Color(0, 120, 255, 255);
            particles.push_back(p);
        }
    }
}

void ParticleSystem::update() {
    for (auto& particle : particles) {
        particle.velocity += particle.force * deltaTime;
        particle.position += particle.velocity * deltaTime;
        
        //... colisiones con bordes
        if (particle.position.x < 0.0f) {
            particle.position.x = 0.0f;
            particle.velocity.x *= -0.5f;
        }
        if (particle.position.x > WINDOW_WIDTH) {
            particle.position.x = WINDOW_WIDTH;
            particle.velocity.x *= -0.5f;
        }
        if (particle.position.y < 0.0f) {
            particle.position.y = 0.0f;
            particle.velocity.y *= -0.5f;
        }
        if (particle.position.y > WINDOW_HEIGHT) {
            particle.position.y = WINDOW_HEIGHT;
            particle.velocity.y *= -0.5f;
        }

        //... colisiones con obstáculos
        for (const auto& obstacle : obstacles) {
            sf::Vector2f obsPos = obstacle.getPosition() + sf::Vector2f(25.0f, 25.0f);
            sf::Vector2f diff = particle.position - obsPos;
            float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);
            if (dist < 25.0f) {
                sf::Vector2f normal = diff / dist;
                particle.position = obsPos + normal * 25.0f;
                
                //... reflexión de velocidad
                float velDotNormal = particle.velocity.x * normal.x + 
                                   particle.velocity.y * normal.y;
                particle.velocity -= 1.8f * velDotNormal * normal;
            }
        }
        
        //... actualización de color basada en velocidad
        float speed = std::sqrt(particle.velocity.x * particle.velocity.x + 
                              particle.velocity.y * particle.velocity.y);
        int blue = static_cast<int>(255 - speed * 5);
        blue = std::max(0, std::min(255, blue));
        particle.color = sf::Color(0, 120, blue, 255);
    }
}

void ParticleSystem::render(sf::RenderWindow& window) {
    sf::CircleShape shape(smoothingLength * 0.5f);
    shape.setOrigin(smoothingLength * 0.5f, smoothingLength * 0.5f);
    
    for (const auto& particle : particles) {
        shape.setPosition(particle.position);
        shape.setFillColor(particle.color);
        window.draw(shape);
    }
    
    for (const auto& obstacle : obstacles) {
        window.draw(obstacle);
    }
}

void ParticleSystem::reset() {
    particles.clear();
    obstacles.clear();
    initializeParticles(WINDOW_WIDTH/4, WINDOW_HEIGHT/4);
    velocityHistory.clear();
    isPaused = false;
}

void ParticleSystem::togglePause() {
    isPaused = !isPaused;
}

void ParticleSystem::handleMouseInput(int x, int y) {
    sf::CircleShape obstacle(25.0f);
    obstacle.setPosition(x - 25.0f, y - 25.0f);
    obstacle.setFillColor(sf::Color(200, 100, 100));
    obstacles.push_back(obstacle);
}

void ParticleSystem::updateStatistics() {
    averageVelocity = 0;
    maxVelocity = 0;
    totalKineticEnergy = 0;
    
    for (const auto& particle : particles) {
        float speed = std::sqrt(particle.velocity.x * particle.velocity.x + 
                              particle.velocity.y * particle.velocity.y);
        averageVelocity += speed;
        maxVelocity = std::max(maxVelocity, speed);
        totalKineticEnergy += 0.5f * particleMass * speed * speed;
    }
    
    if (!particles.empty()) {
        averageVelocity /= particles.size();
    }
    
    //... guardamos historial de velocidades para gráfica
    velocityHistory.push_back(averageVelocity);
    if (velocityHistory.size() > 200) { //... mantener solo últimos 200 frames
        velocityHistory.erase(velocityHistory.begin());
    }
}

const std::vector<Particle>& ParticleSystem::getParticles() const { return particles; }
std::vector<Particle>& ParticleSystem::getParticles() { return particles; }
float ParticleSystem::getSmoothingLength() const { return smoothingLength; }
float ParticleSystem::getParticleMass() const { return particleMass; }
