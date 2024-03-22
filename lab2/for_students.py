from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def create_random(individual_size):
    return [random.choice([True, False]) for _ in range(individual_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def choose_candidates(items, knapsack_max_capacity, population, candidates_number):
    candidates = population[:candidates_number]
    # candidates = random.sample(population, candidates_number)
    candidates_fitness = [fitness(items, knapsack_max_capacity, ind) for ind in candidates]
    return candidates, candidates_fitness


def select_parent(candidates, candidates_fitness):
    total_fitness = sum(candidates_fitness)
    probabilities = [fitness_value / total_fitness for fitness_value in candidates_fitness]
    selected_index = random.choices([i for i in range(len(candidates))], weights=probabilities)[0]
    return candidates[selected_index]


def selection(items, knapsack_max_capacity, population, n_selection):
    selected_parents = []
    candidates, candidates_fitness = choose_candidates(items, knapsack_max_capacity, population, candidates_number)

    for _ in range(n_selection):
        parent1 = select_parent(candidates, candidates_fitness)
        parent2 = select_parent(candidates, candidates_fitness)
        selected_parents.append((parent1, parent2))
    return selected_parents


def crossover(parents, kindermachen_sessions):
    children = []
    for parent1, parent2 in parents:
        # crossover_point = random.randint(1, len(parent1) - 1)
        # child1 = parent1[:crossover_point] + parent2[crossover_point:]
        # child2 = parent2[:crossover_point] + parent1[crossover_point:]
        # children.extend([child1, child2])

        for _ in range(kindermachen_sessions):
            child = []
            for i in range(len(parent1)):
                r = random.randint(0, 1)
                if r:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            children.append(child)
    return children


def mutation(children):
    for child in children:
        for i in range(1):
            mutation_point = random.randint(0, len(child) - 1)
            child[mutation_point] = not child[mutation_point]


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 99

candidates_number = 30
kindermachen_sessions = 2

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
population.sort(key=lambda ind: fitness(items, knapsack_max_capacity, ind), reverse=True)

for _ in range(generations):
    population_history.append(population)

    # DONE: implement genetic algorithm
    selected_parents = selection(items, knapsack_max_capacity, population, n_selection)
    children = crossover(selected_parents, kindermachen_sessions)
    mutation(children)

    combined_population = population + children
    combined_population.sort(key=lambda ind: fitness(items, knapsack_max_capacity, ind), reverse=True)
    population = combined_population[:population_size]
    #add randoms
    randoms = [create_random(len(population[0])) for _ in range(population_size-n_elite)]
    population = population[:-(population_size-n_elite)] + randoms

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
