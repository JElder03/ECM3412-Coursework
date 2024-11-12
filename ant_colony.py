'''
### README ###
NOTE: For accessibility, the graphing aspect of this program has been commented out. If you desire the graphs from ACO1() and
ACO2(), simply uncomment the matplotlib import and the plotting code in ACO1() and ACO2().

To run both the the standard ACO and the greedy heuristic version with the parameters discussed in the report, simply run the code.
Note that this will generate accompanying graphs, terminal outputs, and csv files.

To produce a custom ACO run with no additional output, first remove the calls to ACO1() and ACO2() at the end of this program.
Then, call the bin_packing() function with your desired parameters, including a generator function for the item weight. The
bin_packing()function will return the final solution as a list of bins, as well as the fitness of the solution generated at every 
iteration of the BPP. Note that for custom BPP runs, some parameters which were constant for all BPPs explored, such as the
termination criteria, are constants at the beginning of the program. To edit these, simply change the constant values.
'''

from typing import Callable
'''import matplotlib.pyplot as plt'''
import random
import csv

# Functions for generating item weight from item number
BPP1 = lambda x: x
BPP2 = lambda x: (x * x) / 2
K = 500  # Number of items
NUM_TRIALS = 5  # Number of trials for each parameter set
NUM_EVALUATIONS = 10000 # The number of fitness evaluation before termination

def bin_packing(
    b: int, k: int, p: int, e: float, generator: Callable[[int], list[int]], heuristic: bool = False, h : float = 0
) -> tuple[list[int], list[int|float]]:
    """
    Generates an (approximate) solution to the bin packing problem using ant colony optimisation

    Parameters
    ----------
    b : int
        The number of bins in the bin packing problem
    k : int
        The number of items to put into the bins
    p : int
        The number of ant paths to generate before updating pheromones
    e : float
        The rate of evaporation ever p paths generated
    generator : Callable[[int], list[int]]
        A function for calculating the weight of all k items
    heuristic : bool = False
        A Boolean representing whether to incorporate a greedy heuristic into the ACO
    h : float
        The weight given to the heuristic element of the ACO

    Returns
    -------
    final_solution : list[int]
        The indices, from 0 to b-1, of the nodes visited on the path with the strongest pheromone, including start and end nodes
    progress : list[int|float]
        The quality of solution generated for every iteration of the ACO
    """

    items = []
    for i in range(k):
        items.append(generator(i + 1))

    if heuristic:
        random.shuffle(items)

    # Links from start node
    pheromone = [[[random.uniform(0, 1) for i in range(b)]]]
    # Links from every bin node for every item k
    pheromone = pheromone + [
        [[random.uniform(0, 1) for i in range(b)] for j in range(b)]
        for l in range(k - 1)
    ]
    # Links from final set of bins to end node
    pheromone = pheromone + [[[random.uniform(0, 1)] for i in range(b)]]

    progress = [fitness(get_final_solution(pheromone), b, items)]
    # Iterate the ACO algorithm such that 10,000 fitness tests are completed
    for i in range(int(NUM_EVALUATIONS / p)):
        paths = []
        # Generate ant paths
        for _ in range(p):
            if heuristic:
                paths.append(greedy_ant_path(pheromone, h, items))
            else:
                paths.append(ant_path(pheromone))

        # Update the pheromone proportional to the fitness of each ant path
        for path in paths:
            pheromone = update_pheromone(pheromone, path, items)

        # Evaporate pheromone
        pheromone = [[[link * e for link in bin] for bin in item] for item in pheromone]

        progress.append(fitness(get_final_solution(pheromone), b, items))

    final_solution = get_final_solution(pheromone)
    return final_solution, progress


def ant_path(pheromone: list[list[list[int]]]) -> list[int]:
    """
    Generates an ant path through a graph by recursively selecting links between nodes with a weighted random bias

    Parameters
    ----------
    pheromone : list[list[list[int]]]
        The set of pheromone values for each link, represented in a list containing, for each item to be placed in a bin, a list of containing lists of link weights for each node (bin)
    
    Returns
    -------
    final_solution : list[int]
        The indices of the nodes visited on the ant's path, including start and end nodes
    """

    path = [0]

    for item in pheromone:
        # Get the list of link weights (pheromone) for the links from the current node
        available_nodes = item[path[-1]]
        # Pick a bias random new node to travel to
        path.append(random.choices(range(len(available_nodes)), weights=available_nodes)[0])

    return path

def greedy_ant_path(pheromone: list[list[list[int]]], h : float, items: list[int]) -> list[int]:
    """
    Generates an ant path through a graph by recursively selecting links between nodes with a weighted random bias, with weights incorporating a greedy heuristic.

    Parameters
    ----------
    pheromone : list[list[list[int]]]
        The set of pheromone values for each link, represented in a list containing, for each item to be placed in a bin, a list of containing lists of link weights for each node (bin)
    h : float
        The weight given to the heuristic element of the ACO
    items : list[int]
        The weights of the items to be packed
    
    Returns
    -------
    final_solution : list[int]
        The indices of the nodes visited on the ant's path, including start and end nodes
    """

    path = [0]
    b = len(pheromone[0][0])
    target = sum(items)/b
    bins = [0 for _ in range(b)]

    for i, item in enumerate(pheromone):
        # Get the list of link weights (pheromone) for the links from the current node
        available_nodes = item[path[-1]]

        if i < len(items):
            # add previous item assignment to bin weights
            if i > 0:
                bins[path[-1]] += items[i - 1]
            
            # calculate the distance from the target weight if the item was added to each bin
            dist_from_target = [target-(bin+items[i]) for bin in bins]
            sorted_dist = sorted(set(dist_from_target))
            
            # move negative values to end of sorted list, with most negative last
            neg_index = 0
            while sorted_dist[neg_index]<0:
                neg_index += 1
            for j in range(neg_index, 0, -1):
                sorted_dist.append(sorted_dist[j - 1])
                del sorted_dist[j-1]
            
            # Map each distance to its rank
            rank = [sorted_dist.index(x) + 1 for x in dist_from_target]

            # Add pheromone to each bin's link proportional to its rank
            average_pheromone = sum(available_nodes)/len(available_nodes)
            available_nodes = [available_nodes[j] + (b-rank[j]+1)/b * average_pheromone * h for j in range(b)]
            
        # Pick a bias random new node to travel to
        path.append(random.choices(range(len(available_nodes)), weights=available_nodes)[0])
        
    return path


def fitness(path: list[int], b: int, items: list[int]) -> int | float:
    """
    Calculates the numeric fitness of a given path, where lower values indicate a better solution

    Parameters
    -------
    final_solution : list[int]
        The indices of the nodes visited on the ant's path, including start and end nodes
    b : int
        The number of bins available in the bin packing problem
    items : list[int]
        The weights of the items to be packed

    Returns
    -------
     : int|float
        The fitness of the inputted path, measured as the difference the lightest and heaviest bins
    """

    # Initiate bin weights at 0
    bins = [0 for _ in range(b)]

    # For every node (bin) choice in the path, add the corresponding item weight to the total bin weight
    for i in range(len(path) - 2):
        bin_choice = path[i + 1]
        bins[bin_choice] = bins[bin_choice] + items[i]

    return max(bins) - min(bins)


def update_pheromone(
    pheromone: list[list[list[int]]], path: list[int], items: list[int]
) -> list[list[list[int]]]:
    """
    Takes a path and updates the list of pheromones by adds pheromone to the links used by the path

    Parameters
    -------
    pheromone : list[list[list[int]]]
        The current list of pheromones, by item, then node, then links from that node
    path : list[int]
        The list of nodes, including start and end nodes, in the path being used to update the pheromones
    items : list[int]
        The weights of the items to be packed

    Returns
    -------
    pheromone : list[list[list[int]]]
        The updated list of pheromones, by item, then node, then links from that node
    """

    fitness_value = fitness(path, len(path) - 2, items)
    # Calculate the amount of pheromone to add to the path: 100/fitness
    if fitness_value == 0:
        update = float("inf")  # An unbeatable solution
    else:
        update = 100 / fitness_value

    # Travel the path and add pheromone to every link in the path
    for i in range(len(path) - 1):
        pheromone[i][path[i]][path[i + 1]] += update

    return pheromone


def get_final_solution(pheromone: list[list[list[int]]]) -> list[int]:
    """
    Takes a pheromone list and calculates the solution path by recursively choosing the link with the most pheromone

    Parameters
    -------
    pheromone : list[list[list[int]]]
        The current list of pheromones, by item, then node, then links from that node

    Returns
    -------
    path : list[int]
        The list of nodes, including start and end nodes, in the path identified by ACO as the solution
    """

    path = [0]

    # Start at the start node - index 0
    bin_choice = 0

    # Recursively follow the links with the most weight
    for item in pheromone:
        bin_choice = item[bin_choice].index(max(item[bin_choice]))
        path.append(bin_choice)

    return path

def ACO_1():
    """
    Runs 5 trials of each different combination of p and e on both BPP1 and BPP2. Prints the results of each trial, outputs fitness vs number of iterations graphs for each trial, and save the results in a csv.
    """
    
    problem = 1
    parameters = 1
    progressions = []
    results=[[] for _ in range(NUM_TRIALS + 1)]
    for bpp, b in [(BPP1, 10)]:
        print(f"\n---BPP{problem}---")
        for p, e in [(100, 0.9), (100, 0.6), (10, 0.9), (10, 0.6)]:
            print(f"PARAMETER SET {parameters} (p={p},e={e})")
            results[0].append(f"BPP={problem}, p={p}, e={e}")
            
            for trial in range(NUM_TRIALS):
                solution, progress = bin_packing(b, K, p, e, bpp)
                
                score = fitness(solution, b, [bpp(i + 1) for i in range(K)])
                solution[0] = "start"
                solution[-1] = "end"

                # print(f"Solution: {solution}")
                print(f"Trial {trial + 1} Fitness: {score}")
                
                results[trial + 1].append(score)
                progressions.append(progress)

            parameters += 1
        parameters = 1
        problem += 1

    '''
    for i in range(0, len(progressions), NUM_TRIALS):
        plt.plot(progressions[i])
        plt.xlabel(f"Iteration")
        plt.ylabel("Fitness")
        plt.show()
    '''

    with open(f'BPP_Parameter_Tests.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results)

def ACO_2():
    """
    Test the greedy ACO by running 5 trials for different values of h on BPP1. Prints the results of each trial, outputs fitness vs number of iterations graphs for each first trial, and save the results to a csv.
    """
    
    problem = 1
    parameters = 1
    progressions = []
    results=[[] for _ in range(NUM_TRIALS + 1)]
    for bpp, b in [(BPP1, 10)]:
        print(f"\n---BPP{problem}---")
        for p, e, h in [(10, 0.9, 0.25),(10, 0.9, 0.2),(10, 0.9, 0.15),(10, 0.9, 0.1),(10, 0.9, 0.05),(10, 0.9, 0.0)]:
            print(f"PARAMETER SET {parameters} (p={p},e={e},h={h})")
            results[0].append(f"BPP={problem}, p={p}, e={e}, h={h}")
        
            for trial in range(NUM_TRIALS):
                solution, progress = bin_packing(b, K, p, e, bpp, True, h)
                
                score = fitness(solution, b, [bpp(i + 1) for i in range(K)])
                solution[0] = "start"
                solution[-1] = "end"

                # print(f"Solution: {solution}")
                print(f"Trial {trial + 1} Fitness: {score}")
                
                results[trial + 1].append(score)
                progressions.append(progress)

            parameters += 1
        parameters = 1
        problem += 1

    '''
    for i in range(0, len(progressions), NUM_TRIALS):
        plt.plot(progressions[i])
        plt.xlabel(f"Iteration")
        plt.ylabel("Fitness")
        plt.show()
    '''

    with open(f'BPP_Greedy_Tests.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results)

ACO_1()
ACO_2()