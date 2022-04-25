import numpy as np
import random

# by Martijn Hilders

class Particle:
    # Maximum velocity
    v_max = 0.1
    desired_complexity = None  # the desired complexity value
    # Global best performance (initialized randomly)
    global_best_position = np.array([random.random(), random.random(), random.random(), random.random()])
    global_best_value = 10000

    s = None  # Position (x, y)
    v = None  # Velocity
    f = None  # Performance (complexity at current parameter combination)
    pbest = None  # Previous position of best performance

    complexity_measure = None

    # Particle init
    def __init__(self, position, complexity_function, complexity_desired):

        # Set initial position
        self.s = position
        self.pbest = position

        # Set an initial random speed
        self.v = np.array(
            [random.random() * 0.05, random.random() * 0.05, random.random() * 0.05, random.random() * 0.05])

        # Set complexity measure
        self.complexity_measure = complexity_function

        # intialize desired complexity
        Particle.desired_complexity = complexity_desired

        # Get evaluation at current point
        self.f = self.complexity_measure(self.s)

    # Calculate the next direction, speed, location and performance of the particle
    def next_step(self, a):
        # Create a random number and set b, c to 2 as given in the literature
        R = random.random()
        b, c = 2, 2

        # ** Update functions ** #
        # Calculate the direction and speed based on own personal best and global best
        v_next = a * self.v + b * R * (self.pbest - self.s) + c * R * (Particle.global_best_position - self.s)

        # Cap the velocity at v_max
        for index, velocity in enumerate(v_next):
            if abs(velocity) > Particle.v_max:
                v_next[index] = np.sign(velocity) * Particle.v_max

        self.v = v_next

        # Update the position of the particle
        s_next = self.s + v_next
        self.s = s_next

        # Evaluate the performance of the particle at its new location
        next_f = self.complexity_measure(self.s)

        # Update personal best if value closer to the desired complexity is found!
        if abs(Particle.desired_complexity - next_f) < abs(Particle.desired_complexity - self.f):
            self.pbest = self.s

            # Update the global best
            if abs(Particle.desired_complexity - next_f) < abs(
                    Particle.desired_complexity - Particle.global_best_value):
                Particle.global_best_position = self.s
                Particle.global_best_value = self.complexity_measure(Particle.global_best_position)

        self.f = next_f

    # Get the particle position
    def get_position(self):
        return self.s

    # Get the particle performance at the current location
    def get_performance(self):
        return self.f

    @classmethod
    def get_global_best_val(cls):
        return cls.global_best_value

    @classmethod
    def get_global_best_position(cls):
        return cls.global_best_position

    @classmethod
    def reset_global_best(cls):
        cls.global_best_position = np.array([random.random(), random.random(), random.random(), random.random()])
        cls.global_best_value = None
