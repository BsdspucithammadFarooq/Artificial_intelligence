import random
def initialize_population(pop_size, string_length):
    population = []
    for _ in range(pop_size):
        individual = ''.join(random.choice('01') for _ in range(string_length))
        population.append(individual)
    return population
def calculate_fitness(individual):
    return individual.count('1')
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    parent1 = random.choices(population, probabilities)[0]
    parent2 = random.choices(population, probabilities)[0]
    return parent1, parent2
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring = parent1[:crossover_point] + parent2[crossover_point:]
    return offspring

def mutate(individual, mutation_rate):
    mutated = ''
    for bit in individual:
        if random.random() < mutation_rate:
            mutated += '0' if bit == '1' else '1'  # Flip the bit
        else:
            mutated += bit
    return mutated

def genetic_algorithm(string_length, pop_size, num_generations, mutation_rate):

    population = initialize_population(pop_size, string_length)

    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        new_population = []
        for _ in range(pop_size):
            parent1, parent2 = select_parents(population, fitness_scores)

            offspring = crossover(parent1, parent2)
            offspring = mutate(offspring, mutation_rate)
            new_population.append(offspring)

        population = new_population
        best_individual = max(population, key=calculate_fitness)
        print(f"Generation {generation + 1}: Best String = {best_individual}, Fitness = {calculate_fitness(best_individual)}")
    return max(population, key=calculate_fitness)

if __name__ == "__main__":
    string_length = 10
    pop_size = 20
    num_generations = 50
    mutation_rate = 0.1

    best_solution = genetic_algorithm(string_length, pop_size, num_generations, mutation_rate)
    print(f"Optimal Solution: {best_solution}, Fitness = {calculate_fitness(best_solution)}")
