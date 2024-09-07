import math
import random
from collections import deque

class StentorRoeseli:
    """ Simulates the decision making process inspired by quantum mechanics, Wayfinding theory and Lagrangian Mechanics"""

    def __init__(self, environment_grid, row, col, config=None):
        """
        Initializes the various variables - environmental and organism level
        """
        self.environment_grid = environment_grid
        self.row = row
        self.col = col
        self.environment = environment_grid.grid[row][col]
        default_config = {
            "actions": ["bend", "alternate_cilia", "contract", "detach"],
            "initial_energy": 1000,
            "initial_agentic_energy": 500,
            "initial_systemic_energy": 100,
            "initial_stimulation": 0,
            "uncertainty": 1.0,
            "exploration_rate": 0.96,
            "base_learning_rate": 5.0,
            "learning_rate_adjustment": 0.05,
            "stimulation_threshold_range": (40, 120),
            "initial_sensory_adaptation": 1.0,
            "memory_decay": 0.9,
            "fatigue_increase": 1,
            "fatigue_decrease_rest": 20,
            "energy_increase_rest": 10,
            "base_decision_cost": 5,
            "action_cost_multiplier": 1,
            "dynamic_learning_rate_factor": 0.1,
            "time_step": 1,
            "internal_states": {
                "hunger": 0.5,
                "stress": 0.2,
            },
            
            "memory_size": 100,
            "short_term_memory_size": 10,
            "threshold_adjustment_rate": 0.1,
            "threshold_success_multiplier": 1.1,
            "threshold_failure_multiplier": 0.9,
            
            "sensory_weights": {
                "nutrient_density": 1.0,
                "toxin_level": 2.0,
                "predator_presence": 3.0,
                "light_intensity": 0.5,
                "water_clarity": 0.5,
                "sediment_load": 1.0,
                "algal_bloom_factor": 1.5,
                "bacterial_load": 1.5,
                "temperature": 0.5,
                "salinity": 0.4,
                "ph_level": 0.3
            },
            "random_variation": {
                "learning_rate_variation": 0.2,
                "exploration_rate_variation": 0.1,
                "stimulation_threshold_variation": 0.2,
            },
            "memory_decay_rate": 0.99,
            "short_term_memory_decay_rate": 0.95,
            "action_energy_costs": {
                "bend": 1,
                "alternate_cilia": 1,
                "contract":100,
                "detach": 500,
            },
           "action_context_factors": {
   "bend": {"low_env_score": 4.0, "high_env_score": 3.5}, 
"alternate_cilia": {"low_env_score": 4.0, "high_env_score": 3.5}, 
"contract": {"low_env_score": 3.8, "high_env_score": 4.0}, 
"detach": {"low_env_score": 5.0, "high_env_score": 1.0}

}


,
            "rest_energy_recovery": 10,
            "metabolic_base_cost": 1,
            "sensory_adaptation_rate": 0.98,
            "sensitivity_increase_factor": 1.1,
            "sensitivity_decrease_factor": 0.9,
            "stochastic_event_rate": 0.05,
            "complexity_factor": 0.01,
            "nutrient_boost": 0.2,
            "exploitative_bias": 0.02,
            "explorative_bias": 0.03,
            "temp_sensitivity": 0.1,
            "optimal_temp": 22,
            "salinity_sensitivity": 0.1,
            "optimal_salinity": 0.5,
            "ph_sensitivity": 0.1,
            "optimal_ph": 7.0,
            "base_risk": 0.05,
            "clarity_sensitivity": 0.1,
            "competition_sensitivity": 0.1,
            "symbiosis_sensitivity": 0.1
        }
        
        self.config = default_config if config is None else {**default_config, **config}
        self.actions = self.config["actions"]
        self.energy = self.config["initial_energy"]
        self.agentic_energy = self.config["initial_agentic_energy"]
        self.systemic_energy = self.config["initial_systemic_energy"]
        self.stimulation = self.config["initial_stimulation"]
        self.memory = {action: 0 for action in self.actions}
        self.uncertainty = self.config["uncertainty"]
        self.exploration_rate = self.config["exploration_rate"] * self.config['random_variation']["exploration_rate_variation"]
        self.base_learning_rate = self.config["base_learning_rate"] * self.config['random_variation']["learning_rate_variation"]
        self.learning_rate_adjustment = self.config["learning_rate_adjustment"]
        self.contraction_count = 0
        self.last_action = None
        self.stimulation_threshold = 10000 #random.uniform(*self.config["stimulation_threshold_range"]) * self.config['random_variation']["stimulation_threshold_variation"]
        self.sensory_adaptation = self.config["initial_sensory_adaptation"]
        #self.environment = environment # type: ignore
        self.successful_sequences = []
        self.action_history = []
        self.fatigue = 0
        self.internal_states = self.config["internal_states"].copy()
        self.memory_size = self.config["memory_size"]
        self.short_term_memory = deque(maxlen=self.config["short_term_memory_size"])
        self.long_term_memory = deque(maxlen=self.memory_size)
        self.success_rate = 0
        self.resilience_factor = 1.0
        self.action_probabilities = {action: 1.0 for action in self.actions}

        # Initialize environment-dependent variables
        self.env_state = self.environment.get_state()
        self.toxin_level = self.env_state['toxin_level']
        self.predator_presence = self.env_state['predator_presence']

    def assess_environment(self,env_state=None):
        
        """ Assess the quality of the surrounding environment based on various factors and their optimal values each having different levels of importance"""
        if env_state is None:
            env_state=self.env_state
        #env_state=self.env_state

        impact_factors = {
            "temperature": {"impact_type": "neutral", "importance": 1.0, "optimal_value": 25, "range": 10},
            "ph": {"impact_type": "neutral", "importance": 0.9, "optimal_value": 7, "range": 2},
            "light_intensity": {"impact_type": "positive", "importance": 0.8, "optimal_value": 100, "range": 100},
            "water_clarity": {"impact_type": "positive", "importance": 1.1, "optimal_value": 100, "range": 100},
            "nutrient_quality": {"impact_type": "positive", "importance": 32.5},  # Increased importance
            "toxin_level": {"impact_type": "negative", "importance": 41.3, "range": 1},
            "predator_presence": {"impact_type": "negative", "importance": 1.4},
            "tide": {"impact_type": "neutral", "importance": 0.5},
            "pollution_level": {"impact_type": "negative", "importance": 11.4, "range": 1},
            "oxygen_level": {"impact_type": "positive", "importance": 31.3, "range": 1},
            "microbial_competition": {"impact_type": "negative", "importance": 1.2, "range": 1},
            "symbiotic_factor": {"impact_type": "positive", "importance": 1.0, "range": 1},
            "human_impact_factor": {"impact_type": "negative", "importance": 1.5, "range": 1},
            "climate_change_factor": {"impact_type": "negative", "importance": 1.5, "range": 1},
            "algal_bloom_factor": {"impact_type": "negative", "importance": 1.2, "range": 1},
            "bacterial_load": {"impact_type": "negative", "importance": 1.3, "range": 1},
            "sediment_load": {"impact_type": "negative", "importance": 1.1, "range": 1},
            "macroorganism_competition": {"impact_type": "negative", "importance": 1.3, "range": 1}
        }

        total_score = 0
        total_importance = sum(factor["importance"] for factor in impact_factors.values())
        #print('dsds')
        #print(env_state)

        for factor, value in env_state.items():
            if factor in impact_factors:
                impact = impact_factors[factor]
                importance = impact["importance"]
                impact_type = impact["impact_type"]
                optimal_value = impact.get("optimal_value")
                value_range = impact.get("range", 1)

                if factor == "nutrient_quality":
                    essential_nutrients = ["N", "P", "K"]
                    nutrient_scores = [value.get(nutrient, 0) for nutrient in essential_nutrients]
                    
                    if 0 in nutrient_scores:
                        score = 0
                    else:
                        nutrient_score = (nutrient_scores[0] * nutrient_scores[1] * nutrient_scores[2]) ** (1/3)
                        score = nutrient_score * importance
                elif factor == "predator_presence":
                    total_predator_population = sum(predator["population"] for predator in value)
                    score = (1 - min(total_predator_population / 100, 1)) * importance
                elif factor == "tide":
                    score = importance if value == 'neutral' else 0
                elif isinstance(value, (int, float)):
                    if optimal_value is not None:
                        deviation = abs(value - optimal_value)
                        normalized_deviation = min(deviation / value_range, 1)
                        score = (1 - normalized_deviation) * importance
                    else:
                        normalized_value = min(value / value_range, 1)
                        score = normalized_value * importance if impact_type == "positive" else (1 - normalized_value) * importance

                total_score += score

        normalized_score = total_score / total_importance
        return normalized_score
    

    def calculate_long_term_reward(self, action,state=None):
        """
        Evaluates the long-term reward potential of a given action,
        factoring in environmental opportunities and adaptation strategies.
        """
        long_term_reward = 0
        if state is None:
            state = self.environment.get_state()

     
        nutrient_trend = self.environment.get_trend('nutrient_density')
        oxygen_trend = self.environment.get_trend('oxygen_level')
        symbiosis_trend = self.environment.get_trend('symbiotic_factor')

        if action == "bend" or action == "alternate_cilia":
            nutrient_reward = 0.3 * (nutrient_trend ** 2)
            symbiosis_reward = 0.2 * (symbiosis_trend ** 2)
            long_term_reward += nutrient_reward + symbiosis_reward

        elif action == "contract":
            oxygen_reward = 0.2 * (oxygen_trend ** 2)
            long_term_reward += oxygen_reward

        elif action == "detach":
            exploratory_reward = 0.25 * self.exploration_rate
            long_term_reward += exploratory_reward

        future_potential = 0.15 * self.config["complexity_factor"] * (nutrient_trend + oxygen_trend + symbiosis_trend)
        long_term_reward += future_potential
        long_term_reward = min(1.0, long_term_reward)
        return long_term_reward
    
    def assess_long_term_risk(self, action, current_row, current_col):
        """
        Evaluate the long-term risks associated with an action, particularly detachment,
        by considering the current and neighboring grid environments.
        """
        long_term_risk = 0

        # Evaluate risk based on current environment
        if self.toxin_level > 0.5:
            long_term_risk += 0.2
        if sum(pred["population"] for pred in self.env_state['predator_presence'])> 50:  # Assuming predator presence is a population count
            long_term_risk += 0.3

        # If detachment is the action, assess neighboring grids for additional risk
        if action == "detach":
            neighboring_grids = self.environment_grid.get_adjacent_grids(current_row, current_col)
            for (_,_,grid) in neighboring_grids:
                grid_risk = 0
                grid_state = grid.get_state()
                grid_environment_score = self.assess_environment(grid_state)

                # Threshold for risky environments is taken - 0.3
                if grid_environment_score < 0.3:  
                    grid_risk += 0.3  

                # Assess predator presence in neighboring grid
                grid_predator_presence = sum(pred["population"] for pred in grid_state['predator_presence'])
                if grid_predator_presence > 50:
                    # Increase risk due to predators in the adjacent grid
                    grid_risk += 0.4  

                long_term_risk += grid_risk

        return long_term_risk

    
    def calculate_action_utility(self, action):
        """
        Calculate the utility of a given action, incorporating environmental feedback, context modifiers, and energy costs.
        """
        # Base cost, adjusted to reduce impact without zeroing utility
        base_cost = self.config["action_energy_costs"].get(action, 0) * self.config["action_cost_multiplier"]
        
        # Energy factor: Scale energy contribution and ensure it's positive
        energy_factor = 1 / (1 + base_cost / (self.energy + 1))
        
        # Effectiveness: Normalize effectiveness within a safe range
        memory_sum = sum(self.memory.values()) + len(self.actions)
        effectiveness = (self.memory[action] + 1) / memory_sum  # Keeps effectiveness within (0,1)
        
        # Urgency factor: Ensure urgency has a positive impact, scale to prevent extremes
        urgency_factor = 1 + (self.stimulation / (self.stimulation_threshold + 1))
        
        # Context factor: Environmental context influence
        env_score = self.assess_environment(self.env_state)
        context_modifiers = self.config["action_context_factors"].get(action, {})
        
        if env_score < 0.3:  # Poor environment
            context_factor = context_modifiers.get("low_env_score", 1.0)
        elif env_score > 0.55:  # Favorable environment
            context_factor = context_modifiers.get("high_env_score", 1.0)
        else:
            context_factor = 1.0  # Neutral environment
        
        # Combine factors to calculate utility, ensuring it's positive
        utility = (effectiveness * energy_factor * urgency_factor * context_factor) / (1 + base_cost)
        
        return utility
    
    def dynamic_learning_rate(self, action):
        """
        Calculate a dynamic learning rate based on memory, current stimulation, fatigue, and environmental conditions.
        """
        # Influence of past experiences (memory)
        experience_factor = 1 + (self.memory[action] / (sum(self.memory.values()) + 1))
        
        # Influence of current stimulation (urgency of learning)
        context_factor = 1 + (self.stimulation / self.stimulation_threshold)
        
        # Fatigue reduces the ability to learn effectively
        fatigue_factor = max(0.5, 1 - self.fatigue / 100)
        
        # Adaptation to success rate: increases learning when success rate is high
        success_adjustment = 1 + (self.success_rate - 0.5) * 2 * self.learning_rate_adjustment
        
        # Environmental impact: learning rate adjusts to how well the current environment supports learning
        env_score = self.assess_environment(self.env_state)
        environmental_factor = 1 + (env_score - 0.5) * 2 * self.config.get('env_learning_rate_adjustment', 0.1)
        
        # Calculate the final dynamic learning rate
        learning_rate = (
            self.base_learning_rate 
            * experience_factor 
            * context_factor 
            * fatigue_factor 
            * success_adjustment 
            * environmental_factor
        )
        
        return learning_rate
    
    def detach_action(self):
        """
        Perform the detachment action, choosing the optimal neighboring grid to move to
        based on environmental and risk assessments.
        """
        current_row, current_col = self.row, self.col
        neighboring_grids = self.environment_grid.get_adjacent_grids(current_row, current_col)
        best_grid = None
        lowest_risk = float('inf')

        for (adj_row, adj_col, grid) in neighboring_grids:
            grid_risk = self.assess_long_term_risk("detach", adj_row, adj_col)
            if grid_risk < lowest_risk:
                lowest_risk = grid_risk
                best_grid = (adj_row, adj_col, grid)

        if best_grid:
            # Move to the best grid (update the Stentor's position)
            self.row, self.col, _ = best_grid
            self.env=grid
            self.contraction_count = 0
            self.action_history.append('detach')  # Reset contraction count after detachment
            return "detach"
        else:
            # If no suitable grid is found, default to another action
            return self.decide_action()



    def move_to_grid(self, grid_coords):
        """
        Moves the Stentor to the specified grid and updates the current environment.
        """
        new_row, new_col = grid_coords
        self.row, self.col = new_row, new_col
        self.environment = self.environment_grid.grid[new_row][new_col]

            
    def manage_energies(self):
            """
            Manage the energy levels of the Stentor based on its activities and environmental factors.
            """
            delta_time = self.config["time_step"]

            # Cognitive load influences agentic energy depletion
            cognitive_load = 1 + (self.uncertainty * 0.5)
            agentic_depletion_rate = 2 * cognitive_load
            self.agentic_energy -= agentic_depletion_rate * delta_time

            # Update systemic energy based on recent actions and environmental factors
            metabolic_factor = self.calculate_metabolic_efficiency()
            
            if self.last_action in ["rest", "bend", "alternate_cilia"]:
                self.systemic_energy += 1.5 * delta_time * metabolic_factor
            else:
                self.systemic_energy -= 2 * delta_time * metabolic_factor

            # Update uncertainty dynamically based on recent outcomes
            self.update_uncertainty()
            self.adaptive_agentic_oscillation()

            # Clamp energy levels to ensure they remain within realistic bounds
            self.agentic_energy = max(0, min(self.config["initial_agentic_energy"], self.agentic_energy))
            self.systemic_energy = max(0, min(self.config["initial_systemic_energy"], self.systemic_energy))
            

    def calculate_metabolic_efficiency(self):
        """
        Calculate metabolic efficiency based on environmental conditions like temperature and salinity.
        """
        env_state = self.environment.get_state()
        
        # Temperature effect on metabolic efficiency
        optimal_temp = self.config['optimal_temp']
        temp_efficiency = math.exp(-self.config['temp_sensitivity'] * abs(env_state['temperature'] - optimal_temp))

        # Salinity effect on metabolic efficiency
        optimal_salinity = self.config['optimal_salinity']
        salinity_efficiency = math.exp(-self.config['salinity_sensitivity'] * (env_state['salinity'] - optimal_salinity) ** 2)
        
        # Combine factors to determine overall metabolic efficiency
        metabolic_efficiency = temp_efficiency * salinity_efficiency
        
        return metabolic_efficiency

    def update_uncertainty(self):
        """
        Update uncertainty based on the outcomes of recent decisions and environmental changes.
        """
        # Increase uncertainty if recent outcomes were unpredictable
        if abs(self.stimulation - self.stimulation_threshold) > self.config.get("uncertainty_tolerance", 0.1):
            self.uncertainty = min(1.0, self.uncertainty * 1.1)
        else:
            self.uncertainty = max(0.1, self.uncertainty * 0.9)

    def calculate_decision_cost(self):
        """
        Calculate the cost of making a decision, which reduces the agentic energy.
        The cost is influenced by the uncertainty level, complexity, and current agentic energy.
        """
        base_cost = self.config["base_decision_cost"]
        
        # Incorporate uncertainty: Higher uncertainty increases the decision cost
        uncertainty_factor = 1 + self.uncertainty
        
        # Agentic energy factor: Lower agentic energy increases the cost as decision-making becomes harder
        energy_factor = 1 / (1 + self.agentic_energy / self.config["initial_agentic_energy"])
        
        # Complexity factor: More complex environments and decisions should increase the cost
        complexity_factor = 1 + self.config["complexity_factor"]
        
        # Final decision cost calculation
        decision_cost = base_cost * uncertainty_factor * energy_factor * complexity_factor
        
        # Ensure the cost is within a reasonable range to avoid zero or excessively high costs
        decision_cost = max(0.1, min(decision_cost, self.agentic_energy))
        
        return decision_cost



    def quantum_decide_action(self):
        """
        Quantum-inspired decision-making process with adaptive collapse.
        Actions are placed in superposition states, and the final decision is made by collapsing the state.
        """
        # Initialize superposition state with probability amplitudes
        action_superposition = {action: complex(random.uniform(0, 1), random.uniform(0, 1)) for action in self.actions}
        
        # Apply quantum interference based on environmental factors
        env_state = self.environment.get_state()
        interference_factors = self.calculate_quantum_interference(env_state)
        
        for action in action_superposition:
            # Modify the amplitude based on quantum interference
            action_superposition[action] *= interference_factors.get(action, 1)

        # Collapse the superposition state into a probability distribution
        collapse_probabilities = {action: abs(amplitude)**2 * math.exp(self.agentic_energy) for action, amplitude in action_superposition.items()}
        total_collapse_probability = sum(collapse_probabilities.values())
        normalized_collapse = {action: prob / total_collapse_probability for action, prob in collapse_probabilities.items()}
        
        # Select the action based on collapsed probabilities
        chosen_action = random.choices(list(normalized_collapse.keys()), weights=normalized_collapse.values())[0] 
        
        # Deduct decision-making cost from agentic energy
        self.agentic_energy -= self.calculate_decision_cost()

        return chosen_action

    def calculate_quantum_interference(self, env_state):
        """
        Calculate quantum interference effects based on environmental conditions.
        This simulates the constructive and destructive interference of action probabilities.
        """
        interference_factors = {}

        # Temperature can cause constructive or destructive interference
        temp_factor = math.sin(env_state['temperature'] / self.config['optimal_temp'])
        interference_factors['contract'] = 1 + temp_factor
        interference_factors['detach'] = 1 - temp_factor

        # Salinity can influence the amplitude of certain actions
        salinity_factor = math.cos(env_state['salinity'] / self.config['optimal_salinity'])
        interference_factors['bend'] = 1 + salinity_factor
        interference_factors['alternate_cilia'] = 1 - salinity_factor

        # pH level affects the interference pattern for all actions
        ph_factor = math.exp(-abs(env_state['ph'] - self.config['optimal_ph']) * self.config['ph_sensitivity'])
        for action in self.actions:
            interference_factors[action] = interference_factors.get(action, 1) * ph_factor

        return interference_factors


    def quantum_foresight(self):
        """
        Simulates quantum foresight by predicting potential variability and uncertainty in future environments,
        influencing the risk and reward associated with each action.
        """
        foresight_value = 0

        for factor in ["nutrient_density", "oxygen_level", "symbiotic_factor"]:
            trend = self.environment.get_trend(factor)
            
            # Introduce a quantum-like uncertainty based on environmental trends
            foresight_value += random.uniform(-trend, trend) * self.config["complexity_factor"]

        # The foresight value represents potential variability in future rewards
        return foresight_value





    def long_term_prospection(self):
        """
        Evaluate the potential future benefits and risks associated with different actions.
        This function incorporates environmental trends, quantum foresight, and potential long-term outcomes.
        """
        future_prospects = {}

        for action in self.actions:
            # Immediate utility assessment
            immediate_utility = self.calculate_action_utility(action)

            # Assess the long-term risk associated with this action
            future_risks = self.assess_long_term_risk(action,self.row,self.col)

            # Assess the long-term rewards associated with this action
            future_rewards = self.calculate_long_term_reward(action)

            # Integrate quantum foresight to assess potential future environments
            foresight_reward = self.quantum_foresight()

            # Combine immediate utility, long-term reward, and foresight-adjusted reward, minus future risks
            #print(immediate_utility)
            #print(future_rewards)
            #print(foresight_reward)
            total_prospect_value = immediate_utility + future_rewards + foresight_reward - future_risks

            # Store the total prospective value for this action
            future_prospects[action] = total_prospect_value

        return future_prospects


    def dynamic_prospective_oscillation(self):
        """
        Simulates dynamic quantum-like oscillations in decision-making influenced by agentic energy and environment.
        """
        delta_time = self.config["time_step"]
        
        # Quantum harmonic oscillator model for agentic energy
        omega = 2 * math.pi * self.agentic_energy / self.config["initial_agentic_energy"]
        
        # Map action history to numerical values
        action_values = {
            "bend": 1,
            "alternate_cilia": 1.2,
            "contract": 1.4,
            "detach": 2
        }
        
        # Sum the numerical values of actions in history
        phase_shift = sum(action_values[action] for action in self.action_history) * self.config["complexity_factor"]
        oscillation_factor = math.cos(omega * delta_time + phase_shift)
        
        # Environmental coupling to adjust oscillations
        env_state = self.environment.get_state()
        coupling_factor = 1 + (env_state['temperature'] - self.config['optimal_temp']) * 0.05
        oscillation_factor *= coupling_factor
        
        # Adjust agentic energy based on oscillation
        self.agentic_energy += oscillation_factor * delta_time
        self.agentic_energy = max(0, self.agentic_energy)  # Ensure energy doesn't go negative

        # Generate dynamic prospects influenced by quantum-like oscillations
        prospects = self.long_term_prospection()
        oscillated_prospects = {action: utility + (oscillation_factor * random.uniform(-0.1, 0.1)) 
                                for action, utility in prospects.items()}

        # Adjust with additional risk-reward considerations
        for action in oscillated_prospects:
            risk = self.assess_long_term_risk(action, self.row, self.col)
            reward = self.calculate_long_term_reward(action)
            oscillated_prospects[action] += reward - risk

        return oscillated_prospects

    

    def execute_action(self, action):
        """
        Executes the chosen action and updates internal states and environmental interactions.
        The function integrates the impact of systemic and agentic energy levels.
        """
        # Calculate energy costs and deduct from total energy
        action_cost = self.config["action_energy_costs"].get(action, 0)
        total_energy_cost = action_cost + self.config["metabolic_base_cost"]
        self.energy -= total_energy_cost
        self.agentic_energy -= self.config["base_decision_cost"]
        self.systemic_energy += action_cost

        # Modify internal states based on action type and environment
        if action == "contract":
            self.contraction_count += 1
        elif action == "detach":
            if self.contraction_count == 0:
                action = "contract"
                self.contraction_count += 1
            else:
                return self.detach_action()

        # Update action history and memory
        self.last_action = action
        self.meta_learning(action)
        self.action_history.append(action)
        self.update_success_rate(action)

        self.self_modifying_neural_dynamics(action, outcome=self.success_rate)


        return action

    
    def meta_learning(self, action):
        
        """ 
        Meta learning which aids in the decision making process
        """
        # Use a more dynamic learning rate based on recent success and fatigue
        for act in self.actions:
            learning_adjustment = self.config["dynamic_learning_rate_factor"] * (self.stimulation / self.stimulation_threshold)
            
            # Increase memory for successful actions and decrease for less successful ones
            if act == action:
                self.memory[act] += learning_adjustment
            else:
                self.memory[act] -= learning_adjustment * (1 + self.fatigue / 100)

            # Keep memory values within [0,1] range
            self.memory[act] = max(0, min(1, self.memory[act]))

        # Update memory based on the most recent action
        self.update_memory(action)

    
    def update_memory(self, action):
        """ 
        Updates the short term and long term memory based on recent and previous interaction rates 
        """
        self.short_term_memory = deque(
            [mem * self.config["short_term_memory_decay_rate"] for mem in self.short_term_memory],
            maxlen=self.config["short_term_memory_size"]
        )
        
        self.long_term_memory = deque(
            [mem * self.config["memory_decay_rate"] for mem in self.long_term_memory],
            maxlen=self.memory_size
        )

        # Add a fresh memory for the most recent action
        self.short_term_memory.append(1.0)

        # Reinforce long-term memory if the action was significant
        if action == "contract" or self.stimulation > self.stimulation_threshold:
            self.long_term_memory.append(1.0)


    
    
    def update_success_rate(self, action):
        """
        Updates the success rate based on the outcome of the action, considering environmental factors,
        systemic-agentic energy dynamics, and quantum effects.
        """
        success_delta = 0.1  # Base success adjustment factor

        # Environmental Complexity Impact
        env_score = self.assess_environment()
        complexity_factor = self.config["complexity_factor"] * (1 - env_score)

        # Quantum Influence
        quantum_effect = self.quantum_path_dependency()

        # Success Calculation
        if action == "detach" and self.energy <= 0:
            self.success_rate = max(0, self.success_rate - success_delta)
        else:
            self.success_rate = min(1, self.success_rate + success_delta * (1 + complexity_factor + quantum_effect))

        # Adjust stimulation threshold based on success
        #self.adjust_stimulation_threshold()



    def quantum_path_dependency(self):
        """
        Calculates the quantum path dependency effect based on memory and action history.
        Returns a value that represents the influence of quantum path dependency on decision-making.
        """
        # Introduce decay and amplification based on memory and action history
        decay_factor = self.config["complexity_factor"] * (1 - self.config["memory_decay"])
        path_dependency_effect = 1 + sum(self.memory.values()) * decay_factor

        # Adjust action probabilities based on the accumulated path dependency
        for action in self.actions:
            self.action_probabilities[action] *= path_dependency_effect

        # Normalize the probabilities to ensure they remain within valid bounds
        total_probability = sum(self.action_probabilities.values())
        self.action_probabilities = {action: prob / total_probability for action, prob in self.action_probabilities.items()}

        # Return the path dependency effect for use in other functions
        return path_dependency_effect



    def adjust_stimulation_threshold(self):
        """
        Adjust the stimulation threshold based on the agentic energy levels and success rate and fatigue
        """
        adjustment_rate = self.config["threshold_adjustment_rate"] * (1 + self.uncertainty)

        # Dynamically adjust threshold considering success rate and fatigue
        if self.success_rate > 0.5 and self.fatigue < 0.3:
            self.stimulation_threshold *= (self.config["threshold_success_multiplier"] + adjustment_rate)
        else:
            self.stimulation_threshold *= (self.config["threshold_failure_multiplier"] - adjustment_rate)

        # Apply agentic energy oscillation to the threshold
        oscillation_effect = math.cos(2 * math.pi * self.agentic_energy / self.config["initial_agentic_energy"])
        self.stimulation_threshold += oscillation_effect * adjustment_rate

        # Clamp the threshold between the min and max values
        min_threshold, max_threshold = self.config["stimulation_threshold_range"]
        self.stimulation_threshold = max(min_threshold, min(max_threshold, self.stimulation_threshold))

    def decide_action(self):
        """
        Decides the next action based on current state, long-term prospection, and quantum-inspired decision-making.
        This function prioritizes the decision in the order of Bend, Alternate Cilia, Contract, Detach (BACD),
        while still allowing for some variability based on environmental and internal factors.
        """
        # Update and assess quantum oscillations
        oscillated_prospects = self.dynamic_prospective_oscillation()

        # Assess environment and potential risks
        env_score = self.assess_environment()
        
        
        # Modify the possibility space based on environment, risks, and quantum dynamics
        self.modify_possibility_space()

        # Define a bias order to influence the decision
        action_bias = {"bend": 2.65, "alternate_cilia": 2.5, "contract": 1.1, "detach": 1.0}
        
        # Calculate action utilities using oscillated prospects, environmental assessment, and bias
        action_utilities = {}
        for action in self.actions:
            utility = self.calculate_action_utility(action)
            
            adjusted_utility = (utility + oscillated_prospects.get(action, 0)) * action_bias.get(action, 1)
      
            action_utilities[action] = adjusted_utility

        # Apply quantum decision-making to choose an action based on adjusted utilities
        chosen_action = self.quantum_decide_action()


        # Incorporate environmental feedback and quantum effects for the final action decision
        final_action = self.quantum_feedback_loop(chosen_action)

        # Deduct decision cost from agentic energy
        self.agentic_energy -= self.calculate_decision_cost()

        print(final_action)
        
        return final_action





    
    def modify_possibility_space(self):
        """
        Adjusts the possibility space dynamically based on current environmental conditions,
        internal states, and quantum-inspired oscillations.
        """
        env_state = self.environment.get_state()
        space_modifications = {}

        # Quantum-inspired oscillations affecting all actions
        oscillation_factor = self.dynamic_prospective_oscillation()
        
        # Toxin Level Influence
        toxin_modifier = math.exp(-0.2 * (env_state['toxin_level'] - 0.5) ** 2)
        space_modifications['detach'] = 1.5 * toxin_modifier * oscillation_factor['detach']
        space_modifications['contract'] = 1.3 * toxin_modifier * oscillation_factor['contract']

        # Tide Influence with Quantum Feedback
        if env_state['tide'] == "against":
            space_modifications['detach'] = 0.7 * oscillation_factor['detach']
            space_modifications['contract'] = 1.4 * oscillation_factor['contract']
        elif env_state['tide'] == "towards":
            space_modifications['detach'] = 1.4 * oscillation_factor['detach']
            space_modifications['bend'] = 1.3 * oscillation_factor['bend']

        # Nutrient Density Interaction
        nutrient_modifier = 1 / (1 + math.exp(-0.2 * (env_state['nutrient_density'] - 0.5)))
        space_modifications['bend'] = 1.2 * nutrient_modifier * oscillation_factor['bend']
        space_modifications['alternate_cilia'] = 1.2 * nutrient_modifier * oscillation_factor['alternate_cilia']
        space_modifications['detach'] = 0.7 / nutrient_modifier * oscillation_factor['detach']

        # Predator Presence Factor with Quantum Path Dependency
        total_predator_population = sum(predator["population"] for predator in env_state['predator_presence'])
        predator_modifier = math.exp(-0.3 * (total_predator_population - 0.5) ** 2)
        path_dependency_effect = self.quantum_path_dependency()
        space_modifications['contract'] = 1.4 * predator_modifier * path_dependency_effect
        space_modifications['detach'] = 1.3 * predator_modifier * path_dependency_effect

        # Sediment Load Influence
        sediment_modifier = math.exp(-0.2 * (env_state['sediment_load'] - 0.5) ** 2)
        space_modifications['contract'] = 1.3 * sediment_modifier * oscillation_factor['contract']
        space_modifications['alternate_cilia'] = 0.7 * sediment_modifier * oscillation_factor['alternate_cilia']

        # Adjacent Grid Evaluation for "Detach"
        if 'detach' in space_modifications:
            neighboring_grids = self.environment_grid.get_adjacent_grids(self.row,self.col)
            #print('ree')
            best_grid_score = max(self.assess_environment(grid.get_state()) for (_,_,grid) in neighboring_grids)
            if best_grid_score < 0.4:  # Set a threshold for risk
                space_modifications['detach'] *= 0.5  # Reduce likelihood if adjacent grids are too risky

        # Temperature Influence
        temp_modifier = math.exp(-self.config['temp_sensitivity'] * abs(env_state['temperature'] - self.config['optimal_temp']))
        self.energy *= temp_modifier

        # Salinity Influence
        salinity_modifier = math.exp(-self.config['salinity_sensitivity'] * (env_state['salinity'] - self.config['optimal_salinity']) ** 2)
        for action in self.actions:
            space_modifications[action] = space_modifications.get(action, 1) * salinity_modifier

    # pH Level Influence
        ph_modifier = 1 - self.config['ph_sensitivity'] * abs(env_state['ph'] - self.config['optimal_ph'])
        for action in self.actions:
            space_modifications[action] = space_modifications.get(action, 1) * ph_modifier

        # Water Clarity Influence
        clarity_modifier = self.config['base_risk'] + self.config['clarity_sensitivity'] * (1 / env_state['water_clarity'])
        space_modifications['detach'] *= clarity_modifier

        # Microbial Competition Influence
        competition_modifier = 1 + self.config['competition_sensitivity'] * env_state['microbial_competition']

        # Symbiotic Factor Influence
        symbiosis_modifier = 1 + self.config['symbiosis_sensitivity'] * env_state['symbiotic_factor']
        for action in self.actions:
            space_modifications[action] = space_modifications.get(action, 1) * symbiosis_modifier

        # Apply modifications to action probabilities
        for action in space_modifications:
            self.action_probabilities[action] *= space_modifications[action]

        # Normalize probabilities
        total_probability = sum(self.action_probabilities.values())
        self.action_probabilities = {action: prob / total_probability for action, prob in self.action_probabilities.items()}


    def generate_bead_stimulus(self, speed, number, size):
        """
        Generate the bead stimulus intensity, considering environmental and organism factors.
        """
        env_state = self.environment.get_state()
        
        # Base stimulus from bead properties
        base_stimulus = (speed * 0.4) + (number * 0.35) + (size * 0.25)
        
        # Environmental amplification based on toxin level and predator presence
        total_predator_population = sum(predator["population"] for predator in env_state['predator_presence'])
        environmental_amplification = 1 + (env_state['toxin_level'] * 0.25) + (total_predator_population / 100 * 0.3)
        
        # Interaction with water clarity and nutrient density
        clarity_modifier = (100 / env_state['water_clarity']) ** 0.5
        nutrient_modifier = (env_state['nutrient_density'] + 0.1) ** 0.3
        
        # Adaptive sensitivity based on internal states (fatigue and stress)
        adaptive_sensitivity = 1 + (self.internal_states['stress'] * 0.2) - (self.fatigue * 0.1)
        
        # Total stimulus calculation
        stimulus_intensity = base_stimulus * environmental_amplification * clarity_modifier * nutrient_modifier * adaptive_sensitivity
        return stimulus_intensity






    def respond_to_bead_stimulus(self, speed, number, size):
        """
        Responds to a bead stimulus by evaluating environmental factors, agentic energy, and quantum effects.
        Dynamically chooses the optimal action based on the stimulus characteristics.
        """
        # Generate and process the bead stimulus
        stimulus = self.generate_bead_stimulus(speed, number, size)
        self.stimulation += stimulus * self.sensory_adaptation
        self.energy -= self.config["metabolic_base_cost"]
        self.agentic_energy -= self.config["base_decision_cost"]
        self.uncertainty = max(0.1, self.uncertainty * 0.95)
        self.sensory_adaptation = max(0.5, self.sensory_adaptation * self.config["sensory_adaptation_rate"])
        self.fatigue += self.config["fatigue_increase"]

        # Quantum influence on action choice
        if self.energy <= 0 or self.stimulation > self.stimulation_threshold:
            action = "detach" if self.contraction_count > 0 else "contract"
        else:
            action = self.decide_action()
        
        return self.execute_action(action)

    
    def adaptive_agentic_oscillation(self):

        # Introduce a non-linear oscillation factor that also considers systemic energy
        oscillation_factor = math.cos(2 * math.pi * (self.agentic_energy / self.config["initial_agentic_energy"]) + 
                                    self.systemic_energy / self.config["initial_systemic_energy"])
        
        # Update agentic energy with an adaptive complexity factor
        adaptive_complexity = self.config["complexity_factor"] * (1 + self.uncertainty)
        self.agentic_energy += oscillation_factor * adaptive_complexity
        
        # Ensure energy levels stay within a feasible range
        self.agentic_energy = max(0, min(self.config["initial_agentic_energy"], self.agentic_energy))

    def quantum_feedback_loop(self, initial_decision):
        """
        Refines the action choice based on immediate environmental feedback and quantum interference.
        The function integrates quantum decision-making with real-time environmental conditions.
        """
        # Assess environmental feedback
        env_feedback = self.assess_environment()

        # Adjust action modifiers based on environmental feedback and current stimulation
        action_modifiers = {}
        for action in self.actions:
            modifier = math.exp(-abs(env_feedback - self.stimulation))
            # Apply an additional factor for unpredictability in the environment
            randomness_factor = random.uniform(0.9, 1.1)
            action_modifiers[action] = modifier * randomness_factor

        # Refine the initial decision based on environmental feedback
        if random.random() < action_modifiers.get(initial_decision, 1):
            final_decision = initial_decision
        else:
            # Reconsider the decision with quantum interference, favoring actions with higher modifiers
            weighted_actions = [(action, weight) for action, weight in action_modifiers.items()]
            total_weight = sum(weight for _, weight in weighted_actions)
            probabilities = [weight / total_weight for _, weight in weighted_actions]
            final_decision = random.choices([action for action, _ in weighted_actions], weights=probabilities)[0]

        return final_decision



    
    def self_modifying_neural_dynamics(self, action, outcome):
        """ 
        Based on the recent success and the energy levels updates the neural dynamics 
        """
        # Adaptive adjustment factor with context sensitivity
        adjustment_factor = self.config["dynamic_learning_rate_factor"] * (outcome - self.memory[action])
        
        # Modify memory and action probabilities
        for a in self.actions:
            self.memory[a] += adjustment_factor * (self.stimulation / self.stimulation_threshold)
            self.memory[a] = max(0, min(1, self.memory[a])) 

        # Gradual adjustment to avoid drastic changes, influenced by agentic energy
        adjustment_factor *= self.agentic_energy / self.config["initial_agentic_energy"]
        self.action_probabilities[action] += adjustment_factor
        self.action_probabilities[action] = max(0.1, min(1, self.action_probabilities[action]))  

    

    











    



    
    




    
    
    


