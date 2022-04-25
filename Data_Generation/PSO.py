import random
import numpy as np
from matplotlib import pyplot as plt
from particle import Particle
from complexity_functions import ComplexityFunction
from data_generator import generate_data

# by Martijn Hilders

iterations = 200
numberOfParticles = 100
parameter_range = [[0, 2], [0, 5], [0.01, 1], [0.01, 1]]  # all the ranges for the params
error = 0.008


# suboptimal but must use in form of function for data generator

def PSO(complexity, complexity_function):
    print("Particle Swarm Optimization")

    particles = list()
    complexity = complexity
    complexity_func = complexity_function

    # distribute the particles uniformly random over the search space
    for i in range(numberOfParticles):
        param1 = random.uniform(parameter_range[0][0], parameter_range[0][1])
        param2 = random.uniform(parameter_range[1][0], parameter_range[1][1])
        param3 = random.uniform(parameter_range[2][0], parameter_range[2][1])
        param4 = random.uniform(parameter_range[3][0], parameter_range[3][1])
        position = np.array([param1, param2, param3, param4])
        particle = Particle(position, complexity_function=complexity_func, complexity_desired=complexity)
        particles.append(particle)

    # initialize a
    a = np.linspace(0.9, 0.4, iterations)

    for i in range(iterations):
        for p in particles:
            p.next_step(a[i])

        print(Particle.get_global_best_val())
        if abs(Particle.get_global_best_val() - complexity) < error:
            break

    count = 0
    print()
    print("Best particle position: ", Particle.get_global_best_position())
    print("Best particle value: ", Particle.get_global_best_val())
    for p in particles:
        print("-----")
        print("Particle ", count)
        print("Parameters: ", p.s)
        print("Complexity found: ", p.f)
        count += 1

    # plot the example data x and y
    # plot_data(particles)

    return Particle.get_global_best_val(), Particle.get_global_best_position()
