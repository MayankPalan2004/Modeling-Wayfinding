import random
import math
import numpy as np
import matplotlib.pyplot as plt

class StenorRoeseli:
    """
    Models the Stenor Roeseli organism with attributes and behaviors for responding to environmental stimuli.
    """

    # Class-level attributes for collective memory, action rewards, and neural network weights
    collective_memory = {action: 0 for action in ["bend", "alternate_cilia", "contract", "detach"]}
    action_rewards = {action: 0 for action in ["bend", "alternate_cilia", "contract", "detach"]}
    neural_network_weights = {action: np.random.rand() for action in ["bend", "alternate_cilia", "contract", "detach"]}

    def __init__(self, experiment_day, environment):
        self.actions = ["bend", "alternate_cilia", "contract", "detach"]
        self.energy = 80
        self.stimulation = 0
        self.memory = {action: 0 for action in self.actions}
        self.uncertainty = 1.0
        self.exploration_rate = 0.96
        self.learning_rate = 5.0
        self.contraction_count = 0
        self.last_action = None
        self.stimulation_threshold = random.uniform(40, 120)
        self.sensory_adaptation = 0
        self.environment = environment
        self.experiment_day = experiment_day
        self.successful_sequences = []
        self.action_history = []
        self.fatigue = 0
        self.pre_stimulation_behavior()

    def pre_stimulation_behavior(self):
        """
        Simulates pre-stimulation behavior where the organism randomly performs one of the low-cost actions (bend or alternate_cilia).
        """
        if random.random() < 0.5:
            action = random.choices(["bend", "alternate_cilia"], [1, 1])[0]
            self.execute_action(action)

    def assess_environment(self):
        """
        Assesses the current state of the environment and returns a score based on average environmental conditions.
        Excludes toxin level and predator presence from the score calculation.
        """
        env_state = self.environment.get_state()
        return sum(env_state[:-2]) / len(env_state[:-2])  # Exclude toxin_level and predator_presence from averaging

    def dynamic_learning_rate(self, action):
        """
        Calculates a dynamic learning rate based on the organism's memory of the action and current stimulation.
        """
        experience_factor = 1 + (self.memory[action] / (sum(self.memory.values()) + 1))
        context_factor = 1 + (self.stimulation / self.stimulation_threshold)
        return self.learning_rate * experience_factor * context_factor

    def calculate_action_utility(self, action):
        """
        Calculates the utility of a given action based on various factors including energy, memory, stimulation, and environmental context.
        """
        action_index = self.actions.index(action)
        base_cost = (action_index + 1) ** 2
        energy_factor = math.exp(-base_cost / self.energy)
        effectiveness = (self.memory[action] + StenorRoeseli.collective_memory[action] + 1) / (
            sum(self.memory.values()) + sum(StenorRoeseli.collective_memory.values()) + len(self.actions))
        urgency_factor = 1 + (self.stimulation / self.stimulation_threshold)
        context_factor = 1.0

        env_score = self.assess_environment()
        if env_score < 0.3 and action in ["bend", "alternate_cilia"]:
            context_factor = 4.0 * self.environment.nutrient_density
        elif env_score > 0.7 and action == "contract":
            context_factor = 4.0
        if self.contraction_count == 0 and action in ["bend", "alternate_cilia"]:
            context_factor *= 3.0
        elif self.contraction_count == 0 and action == "contract":
            context_factor *= 0.05

        # Consider toxin level and predator presence
        if self.environment.toxin_level > 0.7 and action == "contract":
            context_factor *= 0.5  # Contracting in a high-toxin environment is less favorable
        if self.environment.predator_presence and action == "detach":
            context_factor *= 2.0  # Detaching is more favorable when a predator is present

        reward_factor = 1 + StenorRoeseli.action_rewards[action]
        neural_factor = StenorRoeseli.neural_network_weights[action]
        return effectiveness * self.stimulation * energy_factor * urgency_factor * context_factor * reward_factor * neural_factor - base_cost

    def meta_learning(self, action):
        """
        Updates the neural network weights for actions based on the recent experience and stimulation.
        """
        for act in self.actions:
            if act == action:
                StenorRoeseli.neural_network_weights[act] += 0.1 * (self.stimulation / self.stimulation_threshold)
            else:
                StenorRoeseli.neural_network_weights[act] -= 0.1 * (self.stimulation / self.stimulation_threshold)
            StenorRoeseli.neural_network_weights[act] = max(0, StenorRoeseli.neural_network_weights[act])

    def prospect_future(self):
        """
        Prospects the utility of future actions considering the current uncertainty.
        """
        prospects = {action: self.calculate_action_utility(action) * (1 - self.uncertainty) for action in self.actions}
        if self.last_action != "contract":
            prospects["detach"] = -float('inf')
        return prospects

    def calculate_decision_cost(self):
        """
        Calculates the cost of making a decision, which increases with uncertainty.
        """
        return 5 * self.uncertainty
    
    def decide_action(self):
        """
        Decides the next action to take based on the current state, action history, and prospect utility.
        """
        if not self.action_history or self.action_history[-1] != "contract":
            if random.random() < 0.9:  # 90% chance to choose bend or alternate_cilia before contract
                return random.choice(["bend", "alternate_cilia"])

        if random.random() < self.exploration_rate * self.uncertainty:
            possible_actions = self.actions[:3] if self.last_action != "contract" else self.actions
            return random.choice(possible_actions)

        prospects = self.prospect_future()
        best_action = max(prospects, key=prospects.get)
        decision_cost = self.calculate_decision_cost()
        self.energy -= decision_cost

        if self.successful_sequences and random.random() < 0.5:
            for seq in self.successful_sequences:
                if self.contraction_count == 0 and "contract" in seq:
                    continue
                if seq and seq[0] in self.actions:
                    best_action = seq[0]
                    self.successful_sequences.remove(seq)
                    break

        return best_action

    def execute_action(self, action):
        """
        Executes the chosen action, updating energy, memory, and other relevant attributes.
        """
        action_index = self.actions.index(action)
        action_cost = (action_index + 1) ** 2
        self.energy -= action_cost
        self.stimulation *= 0.9
        learning_rate = self.dynamic_learning_rate(action)
        self.memory[action] += learning_rate * self.stimulation
        StenorRoeseli.collective_memory[action] += learning_rate * self.stimulation
        self.uncertainty = min(1.0, self.uncertainty * 1.05)

        if action == "contract":
            self.contraction_count += 1
        elif action == "detach":
            if self.contraction_count == 0:
                action = "contract"
                self.contraction_count += 1
            elif random.random() < 0.1:
                action = "contract"
                self.contraction_count += 1

        self.last_action = action
        StenorRoeseli.action_rewards[action] += 0.5
        self.meta_learning(action)

        if self.last_action in ["bend", "alternate_cilia"] and action == "contract":
            self.successful_sequences.append([self.last_action, action])

        self.action_history.append(action)
        return action

    def respond_to_stimulus(self, stimulus):
        """
        Responds to an environmental stimulus by adjusting stimulation and deciding the next action.
        """
        self.stimulation += stimulus * self.sensory_adaptation
        self.energy -= 1
        self.uncertainty = max(0.1, self.uncertainty * 0.95)
        self.sensory_adaptation = max(0.5, self.sensory_adaptation * 0.98)

        if self.energy <= 0 or self.stimulation > self.stimulation_threshold:
            action = "detach" if self.contraction_count > 0 else "contract"
        else:
            action = self.decide_action()
        
        return self.execute_action(action)
    