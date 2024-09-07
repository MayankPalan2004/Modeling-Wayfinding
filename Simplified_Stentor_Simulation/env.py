import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    """
    Simulates the environment in which the Stenor Roeseli organisms live. 
    The environment has attributes such as temperature, pH, light intensity, nutrient density, toxin level, and predator presence.
    """

    def __init__(self):
        # Initialize environmental attributes with random values within specified ranges
        self.temperature = random.uniform(20, 30)
        self.ph = random.uniform(6.5, 8.5)
        self.light_intensity = random.uniform(0, 100)
        self.nutrient_density = random.uniform(0.1, 1.0)
        self.toxin_level = random.uniform(0, 1.0)
        self.predator_presence = random.choice([True, False])

    def update(self):
        """
        Updates the environmental attributes by small random amounts, ensuring they remain within realistic bounds.
        """
        self.temperature += random.uniform(-0.5, 0.5)
        self.ph += random.uniform(-0.1, 0.1)
        self.light_intensity += random.uniform(-5, 5)
        self.nutrient_density += random.uniform(-0.05, 0.05)
        self.toxin_level += random.uniform(-0.1, 0.1)
        self.predator_presence = random.choice([True, False])

        # Ensure attributes stay within realistic limits
        self.temperature = max(15, min(35, self.temperature))
        self.ph = max(5, min(10, self.ph))
        self.light_intensity = max(0, min(100, self.light_intensity))
        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.toxin_level = max(0, min(1, self.toxin_level))

    def get_state(self):
        """
        Returns the current state of the environment as a list of attribute values.
        """
        return [self.temperature, self.ph, self.light_intensity, self.nutrient_density, self.toxin_level, self.predator_presence]