// particle_system.h
#pragma once
#include <vector>
#include <SFML/Graphics.hpp>
#include "constants.h"

struct Particle {
    sf::Vector2f position;
    sf::Vector2f velocity;
    sf::Vector2f force;
    float density;
    float pressure;
    sf::Color color;
};

class ParticleSystem {
private:
    std::vector<Particle> particles;
    std::vector<sf::CircleShape> obstacles;
    float smoothingLength;
    float particleMass;
    float deltaTime;
    bool isPaused;
    float averageVelocity;
    float maxVelocity;
    float totalKineticEnergy;
    int particleCount;
    std::vector<float> velocityHistory;
    

public:
    ParticleSystem(int width, int height) 
        : smoothingLength(15.0f), 
          particleMass(1.0f), 
          deltaTime(1.0f/60.0f) {
        initializeParticles(width/4, height/4);
    }

    void initializeParticles(int startX, int startY);
    void update();
    void render(sf::RenderWindow& window);
    void handleMouseInput(int x, int y);
    const std::vector<Particle>& getParticles() const;
    std::vector<Particle>& getParticles();
    float getSmoothingLength() const;
    float getParticleMass() const;
    const std::vector<float>& getVelocityHistory() const { return velocityHistory; }
    const std::vector<sf::CircleShape>& getObstacles() const { return obstacles; }
    void reset();
    void togglePause();
    void updateStatistics();
    float getAverageVelocity() const { return averageVelocity; }
    float getMaxVelocity() const { return maxVelocity; }
    float getTotalKineticEnergy() const { return totalKineticEnergy; }
    bool getIsPaused() const { return isPaused; }
};

