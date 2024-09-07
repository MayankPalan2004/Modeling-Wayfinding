from env import Environment
from stentor import StenorRoeseli
import random
import math
import numpy as np
import matplotlib.pyplot as plt
def run_experiment(num_organisms=70, max_stimuli=25):
    """
    Runs an experiment simulating the behavior of multiple Stenor Roeseli organisms over several days and stimuli.
    """
    results = []
    environment = Environment()
    for day in range(num_organisms // 10):
        for _ in range(10):
            stentor = StenorRoeseli(experiment_day=day, environment=environment)
            sequence = []
            for _ in range(max_stimuli):
                stimulus = random.uniform(1, 10) * (1 + len(sequence) * 0.15)
                action = stentor.respond_to_stimulus(stimulus)
                sequence.append(action)
                if action == "detach" or stentor.energy <= 0:
                    break
                environment.update()
            results.append(sequence)
    return results

def analyze_results(results):
    """
    Analyzes the results of the experiment, focusing on the frequency and order of specific actions.
    """
    total = len(results)
    contract = sum(1 for seq in results if "contract" in seq)
    detached = sum(1 for seq in results if "detach" in seq)
    c_before_d = sum(1 for seq in results if "contract" in seq and "detach" in seq 
                     and seq.index("contract") < seq.index("detach"))
    ab_before_c = sum(1 for seq in results if ("bend" in seq or "alternate_cilia" in seq) 
                      and "contract" in seq and min(seq.index("bend") if "bend" in seq else float('inf'),
                      seq.index("alternate_cilia") if "alternate_cilia" in seq else float('inf')) 
                      < seq.index("contract"))

    print(f"Total organisms: {total}")
    print(f"Total contract: {contract}")
    print(f"Detached: {detached}")
    print(f"C before D: {c_before_d}/{detached}")
    print(f"A or B before C: {ab_before_c}/{contract} ({ab_before_c/contract*100:.2f}%)")

    for i in range(5):
        remaining = sum(1 for seq in results if len([a for a in seq if a == "contract"]) > i)
        print(f"Remaining after {i+1} contractions: {remaining/total:.2f}")

    action_counts = {action: sum(seq.count(action) for seq in results) for action in ["bend", "alternate_cilia", "contract", "detach"]}
    total_actions = sum(action_counts.values())
    print("\nAction distribution:")
    for action, count in action_counts.items():
        print(f"{action}: {count} ({count/total_actions*100:.2f}%)")

def visualize_results(results):
    """
    Visualizes the results of the experiment using a scatter plot, showing the action sequences of each organism.
    """
    colors = {"bend": "blue", "alternate_cilia": "green", "contract": "red", "detach": "black"}
    plt.figure(figsize=(10, 6))
    for i, sequence in enumerate(results):
        x = range(len(sequence))
        y = [i] * len(sequence)
        c = [colors[action] for action in sequence]
        plt.scatter(x, y, c=c, s=100, alpha=0.6)
    plt.xlabel("Stimulus Count")
    plt.ylabel("Organism Index")
    plt.title("Action Sequences of Stenor Roeseli")
    plt.show()

results = run_experiment()
analyze_results(results)
visualize_results(results)
