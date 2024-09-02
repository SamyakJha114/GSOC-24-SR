import math
import operator
import numpy as np
from deap import base, creator, tools, gp, algorithms
import random

def logabs(x1):
    if x1 == 0:
        return 1
    return math.log(abs(x1))

def protected_div(x1, x2):
    try:
        return x1 / x2 if abs(x2) > 0.001 else 1.
    except ZeroDivisionError:
        return 1.

def protected_exp(x1):
    try:
        return math.exp(x1) if x1 < 100 else 0.0
    except OverflowError:
        return 0.0

def protected_log(x1):
    try:
        return math.log(abs(x1)) if abs(x1) > 0.001 else 0.
    except ValueError:
        return 0.

def protected_sqrt(x1):
    return math.sqrt(abs(x1))

def protected_pow(x1,x2):
    try:
        a = math.pow(x1,x2)
        return a
    except:
        return 1e7
    
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
def make_pset(num_vars):
    pset = gp.PrimitiveSet("MAIN", num_vars)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.tan, 1)
    pset.addPrimitive(abs, 1)
    pset.addPrimitive(math.tanh, 1)
    for i in range(1, 5):
        pset.addTerminal(i)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(protected_pow, 2)
    pset.addPrimitive(protected_exp, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addTerminal(math.pi, name="pi")
    rename_kwargs = {"ARG{}".format(i): f"s_{i+1}" for i in range(0, num_vars)}
    pset.renameArguments(**rename_kwargs)
    return pset

def evalSymbReg(individual, points,toolbox):
    func = toolbox.compile(expr=individual) 
    sqerrors = ((((func(*x) - y)**2)/len(points)) for x, y in points)
    return math.fsum(sqerrors),

def e_lexicase_selection(individuals, k, points):
    selected = []
    for _ in range(k):
        remaining = individuals[:]
        random.shuffle(points)  # Shuffle the test cases
        for point in points:
            errors = [abs(evalSymbReg(ind, [point])[0]) for ind in remaining]
            min_error = min(errors)
            remaining = [ind for ind, error in zip(remaining, errors) if error == min_error]
            if len(remaining) == 1:
                break
        selected.append(random.choice(remaining))
    return selected

    # Seed population with predefined solutions
def seed_population(pop_size,seed_exprs,pset,toolbox):
    population = []
    count = 0
    for expr in seed_exprs:
        try :
            ind = creator.Individual.from_string(expr, pset)
            count += 1
            population.append(ind)
        except :
            continue
    print(len(seed_exprs),count)       
    for _ in range(pop_size - count):
        ind = toolbox.individual()
        population.append(ind)
    return population

def setup_toolbox(pset, points):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", seed_population,toolbox=toolbox)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg, points=points, toolbox=toolbox)
    toolbox.register("select", e_lexicase_selection, toolbox=toolbox)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("map", map)
    
    return toolbox

def run_gp(toolbox, points, seed_expr,pset):
    pop_size = 100
    pop = toolbox.population(pop_size=pop_size, seed_exprs=seed_expr,pset = pset)

    # Evaluate the entire population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind, points=points, toolbox=toolbox)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    ngen = 20
    cxpb, mutpb = 0.5, 0.2

    # Use the eaSimple algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=stats, halloffame=hof, verbose=True)

    print("Best individual:", hof[0])
    print("Fitness:", hof[0].fitness.values)

    # Calculate R2 score
    TSS = 0.0
    mean_y = sum(y for _, y in points) / len(points)
    for _, y in points:
        TSS += (y - mean_y)**2
    print("R2_score:", float(hof[0].fitness.values[0]) / TSS)