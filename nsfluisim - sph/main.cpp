//...... @FECORO, 2023 .......
/*
Código de la simulación de fluidos con SPH (Smoothed Particle Hydrodynamics) en C++ y SFML.

En este código, el fluido se representa como un conjunto de partículas que interactúan entre sí mediante fuerzas de presión y viscosidad. 
La simulación se basa en el método SPH, que es una técnica numérica para simular fluidos mediante partículas discretas.

Básicamente tenemos que el código trabaja como:
- Se inicializan las partículas en una configuración inicial.
- Se calcula la densidad y presión de cada partícula en función de sus vecinos.
- Se calculan las fuerzas de presión y viscosidad que actúan sobre cada partícula.
- Se integran las posiciones y velocidades de las partículas en función de las fuerzas calculadas.
- Se renderiza la simulación en una ventana de SFML.

Este código es solo una implementacón básica y no optimizada de una simulación de fluidos.
Para obtener resultados más realistas y eficientes, se pueden aplicar técnicas avanzadas como:
- Paralelización del cálculo de fuerzas y actualización de partículas.
- Implementación de técnicas de suavizado de densidad y tensión superficial.
- Uso de estructuras de datos espaciales como cuadrículas o árboles para acelerar la búsqueda de vecinos.

Hecho por Felipe Alexander Correa Rodríguez.
*/

//.... main.cpp <- Stack
#include <SFML/Graphics.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

#include "sph_solver.h"
#include "particle_system.h"
#include "constants.h"

class Button {
public:
    Button(const std::string& text, const sf::Vector2f& position, const sf::Vector2f& size) 
        : rect(size) {
        rect.setPosition(position);
        rect.setFillColor(BUTTON_COLOR);
        
        if (!font.loadFromFile("arial.ttf")) {
            //.... usar font del sistema si falla
        }
        
        buttonText.setFont(font);
        buttonText.setString(text);
        buttonText.setCharacterSize(20);
        buttonText.setFillColor(TEXT_COLOR);
        
        //.... centrar el texto
        sf::FloatRect textBounds = buttonText.getLocalBounds();
        buttonText.setPosition(
            position.x + (size.x - textBounds.width) / 2,
            position.y + (size.y - textBounds.height) / 2
        );
    }
    
    bool isMouseOver(const sf::Vector2f& mousePos) const {
        return rect.getGlobalBounds().contains(mousePos);
    }
    
    void draw(sf::RenderWindow& window) {
        window.draw(rect);
        window.draw(buttonText);
    }
    
    void setHovered(bool hovered) {
        rect.setFillColor(hovered ? BUTTON_HOVER_COLOR : BUTTON_COLOR);
    }
    
private:
    sf::RectangleShape rect;
    sf::Text buttonText;
    static sf::Font font;
};

sf::Font Button::font;

int main() {
    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), 
                           "NSFLUID - SPH Simulacion", 
                           sf::Style::Close);
    window.setFramerateLimit(60);
    sf::Vector2i mousePos = sf::Mouse::getPosition(window);
    sf::Vector2f mousePosF(static_cast<float>(mousePos.x), 
                             static_cast<float>(mousePos.y));
    
    ParticleSystem particleSystem(WINDOW_WIDTH, WINDOW_HEIGHT);
    SPHSolver solver;

    //.... variables para fps
    sf::Clock clock;
    sf::Text fpsText;
    sf::Font font;
    if (!font.loadFromFile("Arial.ttf")) {
        //.... usar font del sistema si no encuentra arial
    }
    fpsText.setFont(font);
    fpsText.setCharacterSize(20);
    fpsText.setFillColor(sf::Color::White);
    fpsText.setPosition(10, 10);

    //.... creamos botones
    Button startButton("Start/Pause", sf::Vector2f(10, WINDOW_HEIGHT - 40), sf::Vector2f(100, 30));
    Button resetButton("Reset", sf::Vector2f(120, WINDOW_HEIGHT - 40), sf::Vector2f(100, 30));
    
    //.... texto para estadísticas
    sf::Text statsText;
    statsText.setFont(font);
    statsText.setCharacterSize(16);
    statsText.setFillColor(TEXT_COLOR);
    statsText.setPosition(10, 40);
    
    //.... gráfica de velocidad
    sf::VertexArray velocityGraph(sf::LineStrip);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
            
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    particleSystem.handleMouseInput(event.mouseButton.x, 
                                                  event.mouseButton.y);
                }
            }
        }

        //.... actualización física
        solver.update(particleSystem);
        particleSystem.update();

        //.... cálculo de fps
        float fps = 1.0f / clock.restart().asSeconds();
        fpsText.setString("FPS: " + std::to_string((int)fps));

        //.... actualiza hover de botones
        startButton.setHovered(startButton.isMouseOver(mousePosF));
        resetButton.setHovered(resetButton.isMouseOver(mousePosF));

        //.... actualiza simulación si no está pausada
        if (!particleSystem.getIsPaused()) {
            solver.update(particleSystem);
            particleSystem.update();
        }
        
        //.... actualizamos estadísticas
        particleSystem.updateStatistics();
        
        //.... actualiza texto de estadísticas
        std::stringstream ss;
        ss << "FPS: " << static_cast<int>(1.0f / clock.restart().asSeconds()) << "\n"
           << "Velocidad promedio: " << std::fixed << std::setprecision(2) 
           << particleSystem.getAverageVelocity() << "\n"
           << "Velocidad máxima: " << particleSystem.getMaxVelocity() << "\n"
           << "Energía cinética total: " << particleSystem.getTotalKineticEnergy() << "\n"
           << "Partículas: " << particleSystem.getParticles().size();
        statsText.setString(ss.str());
        
        //.... actualiza gráfica de velocidad
        velocityGraph.clear();
        const auto& history = particleSystem.getVelocityHistory();
        for (size_t i = 0; i < history.size(); ++i) {
            float x = static_cast<float>(WINDOW_WIDTH - 220 + i);
            float y = static_cast<float>(WINDOW_HEIGHT - 100) - history[i] * 2.0f;
            velocityGraph.append(sf::Vertex(sf::Vector2f(x, y), sf::Color::Green));
        }

        //.... renderizado
        window.clear(sf::Color(20, 20, 50));
        particleSystem.render(window);
        window.draw(fpsText);
        window.display();
    }

    return 0;
}

//...............................................| Fin del código |...............................................
// Todos los derechos reservados. @FECORO, 2023.

// Compilo como:
// g++ -std=c++17 -I"C:\msys64\mingw64\include\SFML" -L"C:\msys64\mingw64\lib" -o nsfluidsph main.cpp particle_system.cpp sph_solver.cpp spatial_grid.cpp -lsfml-graphics -lsfml-window -lsfml-system