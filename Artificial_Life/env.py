import random
import math
import copy

class Environment:
    """ The Environment class helps in the creation and simulation of the environment in which the Stentor lives """

    def __init__(self,temperature=None,ph=None, light_intensity=None, nutrient_density=None, nutrient_quality=None, toxin_level=None, predator_presence=None, tide="neutral", season="summer", pollution_level=None, resource_availability=None, oxygen_level=None, water_clarity=None, microbial_competition=None, symbiotic_factor=None, genetic_variability=None, human_impact_factor=None, climate_change_factor=None, salinity=None, algal_bloom_factor=None, bacterial_load=None, sediment_load=None, macroorganism_competition=None):
        """Initializes the various parameters according to the values provided, if not, biologically realistic values are set using random function """

        
        # stentor roeselii found in fresh water regions, although temperature can vary by geographic location and by season, we assume a holistic range to ensure minimum bias 
        if temperature is not None:
            self.temperature=temperature
        else:
            self.temperature=random.uniform(10, 25)



        # Most freshwater habitats have ph close to 7, hence a range chosen between 6.5 to 7.5. This range allows for some natural variation while still remaining within the bounds of what's typically considered healthy for freshwater environments.
        if ph is not None:
            self.ph=ph
        else:
            self.ph=random.uniform(6.5, 8.5)

        
        # Taken on a scale of 100 where 0 means complete darkness and 100 means high light intensity. 
        # The lower end i.e. 0-5 accounts for nighttime conditions
        # The middle range i.e. 20-40 represents the most common conditions where stentor roeselii is active
        # The upper end i.e. 40-60 represents the brightest conditions they typically tolerate.
        # Above 60 is considered generally too harsh 
        if light_intensity is not None:
            self.light_intensity=light_intensity
        else:
            self.light_intensity=random.uniform(0, 60)


        # The environment has nutrients which are essential for survival 
        if nutrient_quality is not None:
            self.nutrient_quality = nutrient_quality
        else:
            self.nutrient_quality = self.generate_nutrient_quality()

        # Calculate the nutrient density based on quality 
        self.nutrient_density=self.calculate_nutrient_density()
        

        # Toxin level in water taken between a scale of 0 and 1 where 0 being no toxin and 1 being the highest level of toxins
        if toxin_level is not None:
            self.toxin_level = toxin_level
        else:
            self.toxin_level = random.uniform(0, 1.0)


        # Adds predators, if required to the environment
        if predator_presence is not None:
            self.predator_presence = predator_presence
        else:
            self.predator_presence=self.generate_predators()

        
        # Pollution level on a scale of 0-1 where 0 is lowest and 1 - highest 
        # Pollution levels depends a lot on regions hence a high degree of variability is to be allowed
        if pollution_level is not None :
            self.pollution_level = pollution_level
        else:
            self.pollution_level = random.uniform(0, 1.0) 

        # Oxygen is one of the most important things for survival
        # Oxygen level taken on a scale of 0 - 1 
        # 0.2 represents the minimum oxygen required for survival 
        if oxygen_level is not None:
            self.oxygen_level = oxygen_level 
        else:
            self.oxygen_level = random.uniform(0.2, 0.8)

        # Water clarity is crucial for aquatic health
        # Generally, in freshwater environments one can expect water clarity between 30 and 70 where 30 indicates high turbidity 
        if water_clarity is not None:
            self.water_clarity = water_clarity
        else:
            self.water_clarity = random.uniform(30, 70)


        # There is competition for resources and resources are generally limited 
        # 0 reflects no competition and 1 represents high competition
        if microbial_competition is not None:
            self.microbial_competition = microbial_competition
        else:
            self.microbial_competition = random.uniform(0, 1.0)

        
        # Symbiotic factor indicates the level of beneficial interactions among organisms
        # The range 0-1.0 is used to represent the absence to the full presence of symbiosis
        if symbiotic_factor is not None:
            self.symbiotic_factor = symbiotic_factor
        else:
            self.symbiotic_factor = random.uniform(0, 1.0)


        # Genetic variability is important for species adaptation and survival
        # A range of 0-1.0 is appropriate to cover the full spectrum of genetic diversity, from low to high
        if genetic_variability is not None:
            self.genetic_variability = genetic_variability
        else:
            self.genetic_variability = random.uniform(0, 1.0)


        # Human impact factor represents the degree of human influence
        # 0 - minimal impact and 1 - serious impact
        if human_impact_factor is not None:
            self.human_impact_factor = human_impact_factor
        else:
            self.human_impact_factor = random.uniform(0, 1.0)
        

        # Climate change factor accounts for the influence of climate change in the aquatic ecosystem
        # 0 - no effect and 1 - severe impact
        if climate_change_factor is not None:
            self.climate_change_factor = climate_change_factor
        else:
            self.climate_change_factor = random.uniform(0, 1.0)


        # Salinity affects the osmoregulation of aquatic organisms and is typically low in freshwater systems
        # 0-2 parts per 1000 is found on average
        if salinity is not None:
            self.salinity = salinity
        else:
            self.salinity = random.uniform(0, 2) 

        
        # Algal bloom factor represents the likelihood of algal blooms, which can disrupt ecosystems
        # The range 0-1.0 reflects no bloom to a severe bloom scenario 
        if algal_bloom_factor is not None:
            self.algal_bloom_factor = algal_bloom_factor
        else:
            self.algal_bloom_factor = random.uniform(0, 1.0)


        # Bacterial load indicates the concentration of bacteria in the environment
        # The range 0-1.0 covers low to high bacterial concentrations, relevant for assessing water quality
        if bacterial_load is not None:
            self.bacterial_load = bacterial_load
        else:
            self.bacterial_load = random.uniform(0, 1.0)


        # Sediment load impacts water quality and habitat conditions
        # The range 0-1.0 represents the spectrum from clear water to high sedimentation 
        if sediment_load is not None:
            self.sediment_load = sediment_load
        else:
            self.sediment_load = random.uniform(0, 1.0)

        
        # Macroorganism competition reflects competition among larger organisms, such as fish or invertebrates.
        # The range 0-1.0 spans from no competition to high competition, affecting species distribution and survival
        if macroorganism_competition is not None:
            self.macroorganism_competition = macroorganism_competition
        else:
            self.macroorganism_competition = random.uniform(0, 1.0)
   
        self.tide = tide
        self.current = random.uniform(0, 1.0)
        self.temperature_trend = 0
        self.time_step = 1
        self.day=0
        self.determine_season()
        self.history = {
    "temperature": [self.temperature],
    "ph": [self.ph],
    "light_intensity": [self.light_intensity],
    "nutrient_density": [self.nutrient_density],
    "nutrient_quality": [self.nutrient_quality],
    "toxin_level": [self.toxin_level],
    "tide": [self.tide],
    "season": [self.season],
    "pollution_level": [self.pollution_level],
    "oxygen_level": [self.oxygen_level],
    "water_clarity": [self.water_clarity],
    "microbial_competition": [self.microbial_competition],
    "symbiotic_factor": [self.symbiotic_factor],
    "genetic_variability": [self.genetic_variability],
    "human_impact_factor": [self.human_impact_factor],
    "climate_change_factor": [self.climate_change_factor],
    "current": [self.current],
    "salinity": [self.salinity],
    "algal_bloom_factor": [self.algal_bloom_factor],
    "bacterial_load": [self.bacterial_load],
    "sediment_load": [self.sediment_load],
    "macroorganism_competition": [self.macroorganism_competition]
}

    def record_history(self):
        """Record the current state into the history for trend analysis."""

        self.history['temperature'].append(self.temperature)
        self.history['ph'].append(self.ph)
        self.history['light_intensity'].append(self.light_intensity)
        self.history['nutrient_density'].append(self.nutrient_density)
        self.history['nutrient_quality'].append(self.nutrient_quality)
        self.history['toxin_level'].append(self.toxin_level)
        self.history['tide'].append(self.tide)
        self.history['season'].append(self.season)
        self.history['pollution_level'].append(self.pollution_level)
        self.history['oxygen_level'].append(self.oxygen_level)
        self.history['water_clarity'].append(self.water_clarity)
        self.history['microbial_competition'].append(self.microbial_competition)
        self.history['symbiotic_factor'].append(self.symbiotic_factor)
        self.history['genetic_variability'].append(self.genetic_variability)
        self.history['human_impact_factor'].append(self.human_impact_factor)
        self.history['climate_change_factor'].append(self.climate_change_factor)
        self.history['current'].append(self.current)
        self.history['salinity'].append(self.salinity)
        self.history['algal_bloom_factor'].append(self.algal_bloom_factor)
        self.history['bacterial_load'].append(self.bacterial_load)
        self.history['sediment_load'].append(self.sediment_load)
        self.history['macroorganism_competition'].append(self.macroorganism_competition)


    def determine_season(self):

        """
        Determine the season based on the day 
        """
        day_in_year = self.day % 360
        
        if 0 <= day_in_year < 90:
            self.season = "spring"
        elif 90 <= day_in_year < 180:
            self.season = "summer"
        elif 180 <= day_in_year < 270:
            self.season = "autumn"
        else:
            self.season = "winter"



    def generate_nutrient_quality(self):
        """
        Generates random values for various nutrients present in the environment.
        This helps the organism in getting an idea about the quality and quantity of the nutrients in the environment.
        """
        def random_nutrient(low, high, probability_of_absence=0.1):
            """
            Generate a random nutrient level, with a chance that the nutrient may be completely absent.
            """
            if random.random() < probability_of_absence:
                return 0.0  # Nutrient is absent
            else:
                return random.uniform(low, high)

        return {
            'C': random_nutrient(0.2, 1.5, probability_of_absence=0.05),  # Carbon source
            'N': random_nutrient(0.05, 1.0, probability_of_absence=0.1),  # Nitrogen source
            'P': random_nutrient(0.01, 0.5, probability_of_absence=0.15),  # Phosphorus
            'K': random_nutrient(0.01, 0.7, probability_of_absence=0.1),  # Potassium
            'Fe': random_nutrient(0.001, 0.05, probability_of_absence=0.2),  # Iron
            'Zn': random_nutrient(0.0001, 0.01, probability_of_absence=0.25),  # Zinc
            'Cu': random_nutrient(0.0001, 0.01, probability_of_absence=0.3)  # Copper
        }
    def calculate_nutrient_density(self):
        # Nutrient density is the average of key macronutrient levels
        key_macronutrients = ['N', 'P', 'K']
        return sum(self.nutrient_quality[n] for n in key_macronutrients) / len(key_macronutrients)
    
    def update_nutrients_on_density(self,factor):
        """ Key macronutrients updates based on change in density """
        key_macronutrients = ['N', 'P', 'K']
        for n in key_macronutrients:
            self.nutrient_quality[n] += factor

    
    


    def generate_predators(self):
        """
        Generate a list of predators with their populations typical for freshwater ecosystems.
        Some predator types may be completely absent.
        """

        predators = []

        def add_predator(type, population_range, probability_of_presence=0.8):
            """
            Conditionally add a predator to the list based on a probability of presence.

            """
            if random.random() < probability_of_presence:
                population = random.randint(*population_range)
                predators.append({"type": type, "population": population})

        # Define the predators with their presence probabilities and population ranges
        add_predator("small crustaceans", (0, 150), probability_of_presence=0.9)
        add_predator("other protozoans", (0, 100), probability_of_presence=0.8)
        add_predator("rotifers", (0, 60), probability_of_presence=0.7)
        add_predator("hydra", (0, 30), probability_of_presence=0.6)
        add_predator("flatworms", (0, 30), probability_of_presence=0.5)
        return predators
    
    
    def check(self):
        self.nutrient_quality['N'] = max(0, min(1, self.nutrient_quality['N']))
        self.nutrient_quality['P'] = max(0, min(1, self.nutrient_quality['P']))
        self.nutrient_quality['K'] = max(0, min(1, self.nutrient_quality['P']))


        
    def update(self, temperature=None, ph=None, light_intensity=None, nutrient_quality=None, toxin_level=None, predator_presence=None, tide=None, season=None, pollution_level=None, resource_availability=None, oxygen_level=None, water_clarity=None, microbial_competition=None, symbiotic_factor=None, genetic_variability=None, human_impact_factor=None, climate_change_factor=None, salinity=None, algal_bloom_factor=None, bacterial_load=None, sediment_load=None, macroorganism_competition=None, time_step=1):
        self.day += time_step
        self.time_step = time_step
        self.determine_season()

        # Daily Updates 
        self.simulate_seasonal_temperature_changes()
        self.check()
        self.update_pollution_level()
        self.check()
        self.update_nutrient_quality()
        self.check()
        self.nutrient_density=self.calculate_nutrient_density()
        self.check()
        self.update_light_intensity()
        self.check()
        self.update_water_clarity()
        self.check()
        self.update_algal_bloom_factor()
        self.check()
        self.simulate_algal_blooms()
        self.check()
        self.update_oxygen_level()
        self.check()
        self.simulate_seasonal_ph_changes()
        self.check()
        self.update_salinity()
        self.check()
    
        
        self.update_microbial_competition()
        self.check()
        self.simulate_microbial_interactions()
        self.check()

        self.update_human_impact_factor()
        self.check()
        self.simulate_human_impact()
        self.check()
        self.simulate_behavioral_factors()
        self.check()


        
        #self.update_salinity()
        
        
   




        if self.day % 30 == 0:
            self.update_climate_change_factor()
            self.check()
            self.simulate_climate_change_effects()
            self.check()
            self.update_genetic_variability()
            self.check()
            self.update_toxin_level()
            self.check()
            self.simulate_toxin_effects()
            self.check()
            self.simulate_stochastic_events()
            self.check()

        if self.day % 7 == 0:
            self.update_bacterial_load()
            self.check()
            self.simulate_bacterial_load()
            self.check()
            self.update_sediment_load()
            self.check()
            self.simulate_sediment_load()
            self.check()
            self.update_symbiotic_factor()
            self.check()
            self.simulate_symbiotic_factor()
            self.check()
            self.update_macroorganism_competition()
            self.check()
            self.simulate_macroorganism_competition()
            self.check()
        self.record_history()



            




        
        
    def simulate_microbial_interactions(self):
        """ Sumulates microbial interactions which can be competitive or symbiotic """
        old=copy.deepcopy(self.nutrient_density)
        if self.microbial_competition > 0.5:
            self.nutrient_density -= 0.003 * self.microbial_competition * self.time_step
            self.oxygen_level -= 0.002 * self.microbial_competition * self.time_step

        if self.symbiotic_factor > 0.5:
            self.nutrient_density += 0.004 * self.symbiotic_factor * self.time_step
            self.oxygen_level += 0.003 * self.symbiotic_factor * self.time_step

        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.update_nutrients_on_density(self.nutrient_density - old)
        self.oxygen_level = max(0, min(1, self.oxygen_level))

        
     
        

    def simulate_algal_blooms(self):
        """ Simulates algal bloom which have adverse effects on the ecosystem """
        if self.algal_bloom_factor > 0.5:
            self.oxygen_level -= 0.005 * self.algal_bloom_factor * self.time_step
            self.water_clarity -= 0.5 * self.algal_bloom_factor * self.time_step
            self.toxin_level += 0.004 * self.algal_bloom_factor * self.time_step

        if self.temperature > 25:
            self.algal_bloom_factor += 0.001 * self.time_step

        if self.nutrient_density > 0.5:
            self.algal_bloom_factor += 0.002 * self.time_step

        self.oxygen_level = max(0, min(1, self.oxygen_level))
        self.water_clarity = max(0, min(100, self.water_clarity))
        self.algal_bloom_factor = max(0, min(1, self.algal_bloom_factor))
        self.toxin_level = max(0, min(1, self.toxin_level))


    def simulate_behavioral_factors(self):
        """
        Simulates the behavioral factors that influence the ecosystem, such as feeding rates,
        movement patterns, and social interactions, and their impact on nutrient density and oxygen levels.
        """
        old = copy.deepcopy(self.nutrient_density )
        # Calculate feeding rate based on nutrient density
        feeding_rate = (0.1 + 0.4 * self.nutrient_density) * self.time_step

        # Movement pattern affecting oxygen levels
        movement_pattern = 0.1 * self.current * self.time_step

        # Impact of feeding and movement on nutrient density and oxygen levels
        self.nutrient_density -= feeding_rate * 0.001
        self.oxygen_level -= movement_pattern * 0.001

        # Social interaction affecting microbial competition
        social_interaction_factor = 0.05 * len(self.predator_presence) * self.time_step
        self.microbial_competition += social_interaction_factor * 0.001

        # Ensure values stay within realistic bounds
        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.update_nutrients_on_density(self.nutrient_density - old)
        self.oxygen_level = max(0, min(1, self.oxygen_level))
        self.microbial_competition = max(0, min(1, self.microbial_competition))



    def simulate_symbiotic_factor(self):
        """
        Simulates the effects of the symbiotic factor on nutrient quality, microbial competition,
        and predator populations within the ecosystem.
        """

        if self.symbiotic_factor > 0.5:
            # Enhance nutrient quality due to beneficial symbiotic relationships (e.g., nitrogen-fixing bacteria)
            self.nutrient_quality['N'] += 0.002 * self.symbiotic_factor * self.time_step
            self.nutrient_quality['P'] += 0.001 * self.symbiotic_factor * self.time_step

            # Reduce microbial competition due to cooperative symbiotic relationships
            self.microbial_competition -= 0.002 * self.symbiotic_factor * self.time_step

            # Improve predator populations due to increased resource availability from symbiosis
            for predator in self.predator_presence:
                predator['population'] *= (1 + 0.005 * self.symbiotic_factor * self.time_step)

        elif self.symbiotic_factor < 0.5:
            # Negative symbiotic relationships (e.g., parasitism) may reduce nutrient quality
            self.nutrient_quality['N'] -= 0.002 * (1 - self.symbiotic_factor) * self.time_step
            self.nutrient_quality['P'] -= 0.001 * (1 - self.symbiotic_factor) * self.time_step

            # Increase microbial competition due to resource strain from parasitic relationships
            self.microbial_competition += 0.002 * (1 - self.symbiotic_factor) * self.time_step

            # Reduce predator populations due to increased mortality in parasitized species
            for predator in self.predator_presence:
                predator['population'] *= (1 - 0.005 * (1 - self.symbiotic_factor) * self.time_step)

    # Ensure values stay within realistic bounds
        self.nutrient_quality['N'] = max(0, min(1, self.nutrient_quality['N']))
        self.nutrient_quality['P'] = max(0, min(1, self.nutrient_quality['P']))
        self.microbial_competition = max(0, min(1, self.microbial_competition))
        for predator in self.predator_presence:
            predator['population'] = max(0, predator['population'])

            



    def simulate_macroorganism_competition(self):
        """
        Simulates the effects of macroorganism competition on the ecosystem, particularly on 
        resource availability, population dynamics, and other relevant factors.
        """
        old=copy.deepcopy(self.nutrient_density)
        # Example effects of increased macroorganism competition:
        if self.macroorganism_competition > 0.5:
            # Decrease resource availability due to competition
            self.nutrient_density -= 0.005 * self.macroorganism_competition * self.time_step

            # Increase stress on the population, possibly reducing genetic variability
            self.genetic_variability -= 0.002 * self.macroorganism_competition * self.time_step

            # Increase predation pressure which might influence predator and prey dynamics
            for predator in self.predator_presence:
                predator['population'] *= 0.98  # Slight reduction due to competition for food

        # Ensure values stay within realistic bounds
        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.update_nutrients_on_density(self.nutrient_density - old)
        self.genetic_variability = max(0, min(1, self.genetic_variability))
        for predator in self.predator_presence:
            predator['population'] = max(0, predator['population'])

    def update_water_clarity(self):
        """ Updates the water quality based on seasonal effect, pollution effect, algal bloom """
        day_in_year = self.day % 360
        seasonal_effect = 5 * math.sin(2 * math.pi * day_in_year / 360)
        fluctuation = random.uniform(-0.5, 0.5) * self.time_step

        # Higher pollution reduces clarity
        pollution_effect = -0.1 * self.pollution_level 

        # Algal blooms reduce clarity
        algal_bloom_effect = -0.05 * self.toxin_level  
        self.water_clarity += fluctuation + seasonal_effect + pollution_effect + algal_bloom_effect
        self.water_clarity = max(0, min(100, self.water_clarity))

    def update_microbial_competition(self):
        """
        Updates the microbial competition based on season, fluctuation, nutrient and pollution 
        """
        # Seasonal effects 
        seasonal_trends = {"spring": 0.002, "summer": 0.005, "autumn": -0.002, "winter": -0.005}
        seasonal_effect = seasonal_trends.get(self.season, 0) * self.time_step

        # Random fluctutation
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        nutrient_effect = sum(self.nutrient_quality.values()) / len(self.nutrient_quality) * 0.005

        # Pollution may inhibit microbial growth
        pollution_stress_effect = -0.002 * self.pollution_level 
        self.microbial_competition += fluctuation + seasonal_effect + nutrient_effect + pollution_stress_effect
        self.microbial_competition = max(0, min(1, self.microbial_competition))



    


    def update_symbiotic_factor(self):
        """
        Updates the symbiotic factor based on various environmental factors - season, toxin , symbiotic
        """

        # Seasonal effects 
        seasonal_trends = {"spring": 0.001, "summer": 0.003, "autumn": -0.001, "winter": -0.002}
        seasonal_effect = seasonal_trends.get(self.season, 0) * self.time_step

        # Random fluctutation
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step

        # Toxins disrupt symbiosis
        toxin_stress_effect = -0.002 * self.toxin_level 

        # Higher clarity and microbial activity support symbiosis
        biodiversity_effect = 0.001 * (self.microbial_competition + self.water_clarity)  
        self.symbiotic_factor += fluctuation + seasonal_effect + toxin_stress_effect + biodiversity_effect
        self.symbiotic_factor = max(0, min(1, self.symbiotic_factor))

    
    def update_oxygen_level(self):
        """
        Updates the oxygen level in the environment based on sinusoidal seasonal trends , temperature and random fluctuations.
        """
         # Determine the day in the year for a complete sinusoidal cycle (assuming a 360-day year for simplicity)
        day_in_year = self.day % 360

        # Sinusoidal variation for oxygen levels
        seasonal_effect = 0.02 * math.sin(2 * math.pi * day_in_year / 360)


        # temperature effect on oxygen solubility
        temp_effect = -0.005 * (self.temperature - 20) / 10  

        # Light intensity drives oxygen production
        photosynthesis_effect = 0.003 * self.light_intensity / 100  

        # Respiration decreases oxygen and as more microbes more respiration
        respiration_effect = -0.002 * self.microbial_competition

        # Base fluctuation due to random environmental factors
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step

        # Update oxygen level with seasonal,respiration and photosynthesis effect and random fluctuation
        self.oxygen_level += fluctuation + seasonal_effect + temp_effect + photosynthesis_effect + respiration_effect

         # Ensure oxygen levels stay within realistic bounds
        self.oxygen_level = max(0, min(1, self.oxygen_level))



    

   
    


    def update_pollution_level(self):
        """
        Updates the pollution level in the environment based on seasonal trends and random fluctuations.
        """

        # Base fluctuation due to random environmental factors
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step

        # Seasonal adjustments - https://www.researchgate.net/publication/347418013_Investigation_of_water_quality_in_wet_and_dry_seasons_under_climate_change
        seasonal_adjustments = {
            "spring": 0.01,  # Increase due to higher runoff carrying pollutants into water bodies
            "summer": 0.005, # Potential increase due to concentration of pollutants with lower water levels
            "autumn": 0.003, # Slight decrease as some pollutants may settle or degrade, but can still concentrate
            "winter": 0.01   # Increase due to rainfall and runoff carrying pollutants
        }

        # Get the seasonal adjustment for the current season
        seasonal_effect = seasonal_adjustments.get(self.season, 0) * self.time_step
        
        # nutrient runoff effect on pollution 
        nutrient_runoff_effect = 0.002 * (self.nutrient_quality['N'] + self.nutrient_quality['P']) / 2

        # Update pollution level with seasonal and nutrient runoff effect and random fluctuation
        
        self.pollution_level += fluctuation + seasonal_effect + nutrient_runoff_effect

        # Ensure pollution levels stay within realistic bounds 
        self.pollution_level = max(0, min(1, self.pollution_level))

    

    

    def update_toxin_level(self):
        """
        Updates the toxin level based on seasonal trends and random fluctuations.
        """
        # Base fluctuation due to random environmental factors
        fluctuation = random.uniform(-0.01, 0.01) * self.time_step

        # Seasonal adjustments - (https://www.sciencedirect.com/science/article/pii/S2214158820300192)
        seasonal_adjustments = {
            "spring": 0.005,  # Increase due to higher temperatures and nutrient runoff
            "summer": 0.01,   # Further increase due to peak temperatures and algal blooms
            "autumn": -0.003, # Slight decrease as temperatures cool
            "winter": -0.005  # Decrease due to low temperatures and reduced biological activity
        }

        # Get the seasonal adjustment for the current season
        seasonal_effect = seasonal_adjustments.get(self.season, 0) * self.time_step

        # Update toxin level with seasonal effect and random fluctuation
        self.toxin_level += fluctuation + seasonal_effect

        # High microbial competition could either mitigate or exacerbate toxin levels depending on the dominant microbial processes
        microbe_effect = -0.002 * self.microbial_competition
        self.toxin_level += microbe_effect


        # Warmer temps + nutrients -> blooms
        algal_bloom_effect = 0.005 * self.nutrient_quality['P'] * self.temperature / 30  

        self.toxin_level+=algal_bloom_effect

        # Ensure toxin levels stay within realistic bounds 
        self.toxin_level = max(0, min(1, self.toxin_level))

    def simulate_toxin_effects(self):
        """
        Simulates the effects of toxin levels on various ecosystem factors, including microbial competition,
        oxygen levels, population health, and nutrient cycling.
        """

        if self.toxin_level > 0.5:
            # Decrease oxygen levels due to the toxic impact on aquatic life and microbial activity
            self.oxygen_level -= 0.004 * self.toxin_level * self.time_step

            # Increase microbial competition as some species might become more dominant
            self.microbial_competition += 0.003 * self.toxin_level * self.time_step

            # Decrease nutrient quality as toxins can inhibit nutrient cycling processes
            self.nutrient_quality['N'] -= 0.002 * self.toxin_level * self.time_step
            self.nutrient_quality['P'] -= 0.001 * self.toxin_level * self.time_step

            # Impact on population health
            
            for predator in self.predator_presence:
                predator['population'] *= (1 - 0.005 * self.toxin_level * self.time_step)

        # Ensure values stay within realistic bounds
        self.oxygen_level = max(0, min(1, self.oxygen_level))
        self.microbial_competition = max(0, min(1, self.microbial_competition))
        self.nutrient_quality['N'] = max(0, min(1, self.nutrient_quality['N']))
        self.nutrient_quality['P'] = max(0, min(1, self.nutrient_quality['P']))
        for predator in self.predator_presence:
            predator['population'] = max(0, predator['population'])



    


    def update_nutrient_quality(self):
        """
        Updates nutrient quality by applying small fluctuations and seasonal trends.
        Ensures nutrient levels stay within realistic bounds.
        """

        # Seasonal trends for each nutrient 
        seasonal_trends = {
            "spring": {"C": 0.003, "N": 0.002, "P": 0.004, "Fe": 0.001, "Zn": 0.001, "Cu": 0.001},
            "summer": {"C": 0.001, "N": -0.001, "P": -0.002, "Fe": -0.001, "Zn": -0.001, "Cu": -0.001},
            "autumn": {"C": -0.001, "N": 0.002, "P": 0.003, "Fe": 0.001, "Zn": 0.001, "Cu": 0.001},
            "winter": {"C": -0.002, "N": -0.002, "P": -0.001, "Fe": 0.000, "Zn": 0.000, "Cu": 0.000}
        }

        # Get seasonal trends for the current season
        trends = seasonal_trends.get(self.season, {})

        for nutrient in self.nutrient_quality:
            # Apply a small random fluctuation
            fluctuation = random.uniform(-0.005, 0.005) * self.time_step

            # Apply seasonal trend if available 
            seasonal_adjustment = trends.get(nutrient, 0) * self.time_step

            # Microbes deplete nutrients
            uptake_effect = -0.001 * self.microbial_competition  

            # Update nutrient quality
            self.nutrient_quality[nutrient] += fluctuation + seasonal_adjustment + uptake_effect

            # Ensure nutrient levels stay within realistic bounds (0 to 1)
            self.nutrient_quality[nutrient] = max(0, min(1, self.nutrient_quality[nutrient]))

        return self.nutrient_quality
    

   


    def update_light_intensity(self):
        """
        Updates the light intensity based on the season and time step, with added random daily fluctuations.
        """

        # Seasonal base light intensity 
        seasonal_light_base = {
            "spring": 70,  # Moderate light intensity
            "summer": 90,  # High light intensity
            "autumn": 60,  # Moderate light intensity
            "winter": 40   # Lower light intensity
        }

        # Get base light intensity based on current season and if season not defined, defaults to 60
        base_intensity = seasonal_light_base.get(self.season, 60)  

        # Add a daily fluctuation to simulate weather variations 
        fluctuation = random.uniform(-0.5, 0.5) * self.time_step
         
        # Get the effect of water clarity on light intensity
        water_clarity_effect = 0.05 * (self.water_clarity - 50) / 100

        self.light_intensity = base_intensity + fluctuation + water_clarity_effect

        # Ensure light intensity stays within realistic bounds (0 to 100%)
        self.light_intensity = max(0, min(100, self.light_intensity))


    

    

    def simulate_seasonal_ph_changes(self):
        """
        Simulates daily changes in pH based on the season 
        The pH is adjusted considering the effects of temperature and seasonal variations
        """

        # Sinusoidal pH change over a year (360 days) to simulate seasonal variation
        day_in_year = self.day % 360

        # Apply the seasonal effect
        seasonal_effect = 0.05 * math.sin(2 * math.pi * day_in_year / 360)

        # Feedback loop effect - temperature, oxygen, photosynthesis and microbial
        temp_effect = -0.005 * (self.temperature - 20) / 10
        oxygen_effect = 0.002 * (self.oxygen_level - 0.8)
        photosynthesis_effect = 0.003 * self.light_intensity / 100
        microbial_activity_effect = -0.002 * self.microbial_competition
        self.ph += seasonal_effect + temp_effect + oxygen_effect + photosynthesis_effect + microbial_activity_effect

        # Ensure pH remains within realistic bounds 
        self.ph = max(6.0, min(9.0, self.ph))


     



    def simulate_seasonal_temperature_changes(self):
        """
        Simulates daily temperature changes with sinusoidal seasonal variation 
        """
        # We assume 360 days a year and 4 seasons each lasting 90 days
        day_in_year = self.day % 360

        # We set the mean temperatures for each season
        seasonal_mean_temperatures = {
            "spring": 20,
            "summer": 28,
            "autumn": 22,
            "winter": 16
        }

        # We set amplitude of temperature change within the season
        seasonal_amplitude = {
            "spring": 5,
            "summer": 7,
            "autumn": 5,
            "winter": 4
        }

        # We calculate the seasonal component of temperature
        if self.season == "spring":
            self.temperature_trend = seasonal_mean_temperatures["spring"] + seasonal_amplitude["spring"] * math.sin(math.pi * (day_in_year % 90) / 90)
        elif self.season == "summer":
            self.temperature_trend = seasonal_mean_temperatures["summer"] + seasonal_amplitude["summer"] * math.sin(math.pi * (day_in_year % 90) / 90)
        elif self.season == "autumn":
            self.temperature_trend = seasonal_mean_temperatures["autumn"] + seasonal_amplitude["autumn"] * math.sin(math.pi * (day_in_year % 90) / 90)
        elif self.season == "winter":
            self.temperature_trend = seasonal_mean_temperatures["winter"] + seasonal_amplitude["winter"] * math.sin(math.pi * (day_in_year % 90) / 90)

        # We add day-to-day random fluctuations
        daily_fluctuation = random.uniform(-0.5, 0.5)  

        # We update temperature based on seasonal trend and daily fluctuation
        self.temperature = self.temperature_trend + daily_fluctuation

        # Ensure temperature remains within realistic bounds for the particular season
        if self.season == "summer":
            self.temperature = max(25, min(35, self.temperature))
        elif self.season == "winter":
            self.temperature = max(10, min(20, self.temperature))
        else:  # Spring and Autumn
            self.temperature = max(15, min(30, self.temperature))


    def update_sediment_load(self):

        """
        Updates the sediment load factor based on runoff, erosion and random fluctuation
        """
        # Pollution often correlates with runoff
        runoff_effect = 0.01 * self.pollution_level 

        # Low clarity suggests high sediment
        erosion_effect = 0.008 * (1 - self.water_clarity / 100)  

        # Random fluctuation added
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        self.sediment_load += fluctuation + runoff_effect + erosion_effect

        # Ensuring the sediment load stays bounded
        self.sediment_load = max(0, min(1, self.sediment_load))

    def simulate_sediment_load(self):
        """ 
        Simulates the effect of sediment load 
        """
        old=copy.deepcopy(self.nutrient_density)
        if self.sediment_load > 0.5:
            self.water_clarity -= 0.5 * self.sediment_load * self.time_step
            self.nutrient_density -= 0.002 * self.sediment_load * self.time_step
        

        self.water_clarity = max(0, min(100, self.water_clarity))
        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.update_nutrients_on_density(self.nutrient_density - old)



    def update_bacterial_load(self):

        """
        Updates bacterial load based on nutrient effect, temperature effect and oxygen effect 
        """
        nutrient_effect = 0.006 * sum(self.nutrient_quality.values()) / len(self.nutrient_quality)
        temp_effect = 0.004 * (self.temperature - 20) / 10

         # Low oxygen boosts anaerobic bacteria
        oxygen_effect = -0.003 * (self.oxygen_level - 0.8) 

        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        self.bacterial_load += fluctuation + nutrient_effect + temp_effect + oxygen_effect
        self.bacterial_load = max(0, min(1, self.bacterial_load))

    def simulate_bacterial_load(self):
        """ 
        Simulates the effect of bacterial load 
        """

        old=copy.deepcopy(self.nutrient_density)

        if self.bacterial_load > 0.5:
            self.oxygen_level -= 0.005 * self.bacterial_load * self.time_step
            self.nutrient_density -= 0.003 * self.bacterial_load * self.time_step

        if self.algal_bloom_factor > 0.5:
            self.bacterial_load += 0.001 * self.time_step

        self.oxygen_level = max(0, min(1, self.oxygen_level))
        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.update_nutrients_on_density(self.nutrient_density - old)
        self.bacterial_load = max(0, min(1, self.bacterial_load))


    def update_algal_bloom_factor(self):

        """ 
        Update the algal bloom factor based on nutrient , temperature and random daily fluctuation
        """
        nutrient_effect = 0.007 * (self.nutrient_quality['P'] + self.nutrient_quality['N']) / 2
        temperature_effect = 0.005 * (self.temperature - 20) / 10
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        self.algal_bloom_factor += fluctuation + nutrient_effect + temperature_effect
        self.algal_bloom_factor = max(0, min(1, self.algal_bloom_factor))


    def update_salinity(self):

        """
        Updates the salinity based on evaporation and freshwater effect alongside random daily fluctuations 
        """
        # Warmer temps increase evaporation
        evaporation_effect = 0.01 * (self.temperature - 20) / 10

        # Clearer water indicates more freshwater
        freshwater_input_effect = -0.005 * (self.water_clarity / 100)  
        fluctuation = random.uniform(-0.05, 0.05) * self.time_step
        self.salinity += fluctuation + evaporation_effect + freshwater_input_effect

        # Adjusted for freshwater range
        self.salinity = max(0, min(5, self.salinity))  

    def update_climate_change_factor(self):
        """
        Updates the climate change factor based on co2 effect , human effect and random daily fluctuation
        """

        co2_effect = 0.004 * self.pollution_level

        # More human impact means less mitigation
        mitigation_effect = -0.002 * (1 - self.human_impact_factor) 
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        self.climate_change_factor += fluctuation + co2_effect + mitigation_effect
        self.climate_change_factor = max(0, min(1, self.climate_change_factor))

    def simulate_climate_change_effects(self):
        """ 
        Climate changes adversely affects many environmental factors
        """
        old=copy.deepcopy(self.nutrient_density)
        self.temperature_trend += 0.001 * self.climate_change_factor * self.time_step
        self.water_clarity -= 0.01 * self.climate_change_factor * self.time_step
        self.nutrient_density -= 0.005 * self.climate_change_factor * self.time_step
        self.ph -= 0.002 * self.climate_change_factor * self.time_step

        self.temperature = max(15, min(35, self.temperature_trend * self.time_step / 90 + self.temperature))
        self.water_clarity = max(0, min(100, self.water_clarity))
        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.update_nutrients_on_density(self.nutrient_density - old)
        self.ph = max(6.0, min(9.0, self.ph))



    def update_human_impact_factor(self):
        """
        Updates the human impact based on pollution effect , conservation effect and random daily fluctuation
        """

        pollution_effect = 0.005 * self.pollution_level

        # High symbiosis might indicate successful conservation
        conservation_effect = -0.003 * self.symbiotic_factor  

        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        self.human_impact_factor += fluctuation + pollution_effect + conservation_effect
        self.human_impact_factor = max(0, min(1, self.human_impact_factor))

    def simulate_human_impact(self):
        """ 
        Simulates the human impact based on the human impact factor 
        """
        if self.human_impact_factor > 0.5:
            self.toxin_level += 0.01 * self.human_impact_factor * self.time_step
            self.pollution_level += 0.01 * self.human_impact_factor * self.time_step
            self.microbial_competition += 0.003 * self.human_impact_factor * self.time_step
            self.sediment_load += 0.002 * self.human_impact_factor * self.time_step

        self.toxin_level = max(0, min(1, self.toxin_level))
        self.pollution_level = max(0, min(1, self.pollution_level))
        self.microbial_competition = max(0, min(1, self.microbial_competition))
        self.sediment_load = max(0, min(1, self.sediment_load))


    def update_genetic_variability(self):
        """
        Update genetic variabilty based on stress and mutation
        """
        stress_effect = -0.002 * (self.pollution_level + self.climate_change_factor)

         # Less human impact allows more natural mutation
        mutation_effect = 0.003 * (1 - self.human_impact_factor) 
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        self.genetic_variability += fluctuation + stress_effect + mutation_effect
        self.genetic_variability = max(0, min(1, self.genetic_variability))

    def update_macroorganism_competition(self):
        """ 
        Update the macroorganism competition based on pollution, temperature and human impact factor
        """
    
        # Random daily fluctuation
        fluctuation = random.uniform(-0.005, 0.005) * self.time_step
        
        # Environmental stressors affecting competition
        pollution_effect = -0.002 * self.pollution_level

        # Warmer temperatures may increase competition
        temperature_effect = 0.003 * (self.temperature - 20) / 10  
        human_impact_effect = -0.002 * self.human_impact_factor
        
        self.macroorganism_competition += fluctuation + pollution_effect + temperature_effect + human_impact_effect
        
        self.macroorganism_competition = max(0, min(1, self.macroorganism_competition))

    def simulate_stochastic_events(self):
        """ 
        Simulates events which occur rarely but can have drastic effects 
        """
        event_probability = random.random()
        old=copy.deepcopy(self.nutrient_density)

        if event_probability < 0.01:
            # Storm event
            self.current += 0.1
            self.sediment_load += 0.02
            self.water_clarity -= 1
            self.pollution_level += 0.01
            print("A storm has occurred, increasing current and sediment load, and decreasing water clarity.")

        elif event_probability < 0.02:
            # Sudden pollution spike
            self.toxin_level += 0.02
            self.pollution_level += 0.02
            self.nutrient_density -= 0.01
            print("A sudden pollution spike has occurred, increasing toxin and pollution levels.")

        elif event_probability < 0.03:
            # Seasonal migration
            migrating_predators = random.choice(self.predator_presence)
            migrating_predators['population'] *= 0.5
            print(f"Seasonal migration occurred, reducing population of {migrating_predators['type']} by half.")

        elif event_probability < 0.035:
            # Disease outbreak
            self.nutrient_density -= 0.02
            self.oxygen_level -= 0.01
            print("A disease outbreak has occurred, reducing resource availability and oxygen levels.")

        elif event_probability < 0.04:
            # Reproductive cycle
            for predator in self.predator_presence:
                predator['population'] *= 1.5
            print("A reproductive cycle has occurred, increasing predator populations.")

        elif event_probability < 0.045:
            # Fishing event
            fishing_target = random.choice(self.predator_presence)
            fishing_target['population'] *= 0.7
            print(f"Fishing event has occurred, reducing population of {fishing_target['type']} by 30%.")

        elif event_probability < 0.05:
            # Habitat restoration
            self.water_clarity += 1
            self.nutrient_density += 0.01
            self.pollution_level -= 0.01
            print("Habitat restoration effort has occurred, improving water clarity and resource availability, and reducing pollution levels.")

        # Ensure all values stay within realistic bounds
        self.current = max(0, min(1, self.current))
        self.sediment_load = max(0, min(1, self.sediment_load))
        self.water_clarity = max(0, min(100, self.water_clarity))
        self.pollution_level = max(0, min(1, self.pollution_level))
        self.toxin_level = max(0, min(1, self.toxin_level))
        self.nutrient_density = max(0, min(1, self.nutrient_density))
        self.update_nutrients_on_density(self.nutrient_density - old)
        self.oxygen_level = max(0, min(1, self.oxygen_level))

    def get_state(self):

        """ 
        Returns the current state of the environment
        """
        return {
            "temperature": self.temperature,
            "ph": self.ph,
            "light_intensity": self.light_intensity,
            "nutrient_density": self.nutrient_density,
            "nutrient_quality": self.nutrient_quality,
            "toxin_level": self.toxin_level,
            "predator_presence": self.predator_presence,
            "tide": self.tide,
            "season": self.season,
            "pollution_level": self.pollution_level,

            "oxygen_level": self.oxygen_level,
            "water_clarity": self.water_clarity,
            "microbial_competition": self.microbial_competition,
            "symbiotic_factor": self.symbiotic_factor,
            "genetic_variability": self.genetic_variability,
            "human_impact_factor": self.human_impact_factor,
            "climate_change_factor": self.climate_change_factor,
            "current": self.current,
            "salinity": self.salinity,
            "algal_bloom_factor": self.algal_bloom_factor,
            "bacterial_load": self.bacterial_load,
            "sediment_load": self.sediment_load,
            "macroorganism_competition": self.macroorganism_competition
        }
    def get_trend(self, factor):
        """Calculate the trend for a given environmental factor."""

        # Not enough data to determine a trend
        if factor not in self.history or len(self.history[factor]) < 2:
            return 0  
        
        # Analyze the last 5 entries
        recent_values = self.history[factor][-5:]  
        trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        return trend
    
    def predict_future_state(self):
        """
        Predicts the future state of the environment based on current trends in the environmental factors.
        Applies trend data to project future conditions.
        """
        future_state = {}

       
        for factor, value in self.get_state().items():
            if isinstance(value, (int, float)):
                trend = self.get_trend(factor)

                # Project future state based on trend
                future_value = value + trend  

                 # Ensure non-negative values
                future_state[factor] = max(0, future_value) 
            else:
                # Non-numeric factors are assumed to stay the same or can be further processed if necessary
                future_state[factor] = value
        
        return future_state












    


        

          


          

        


        

        
