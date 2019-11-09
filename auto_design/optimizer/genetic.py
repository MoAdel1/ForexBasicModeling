#%% code imports
import random
import numpy as np
from deap import base, creator, tools, algorithms
from auto_design.feature_engineering.encoder import build_encoder
from auto_design.modeling.neural_nets import build_model


#%% functions definition
def eval_fit(individual, data):
    '''function to train and evaluate the fitness of the pipeline on the given data

    Parameters
    ----------
    individual : list 
    data : dict

    Returns
    -------
    fitness : float
        the loss value on the fitness function
    '''
    assert len(individual)==7, 'length of soultion must be 7'
    # extract the train, validate and test data
    x_train = data['x_train'] 
    y_train = data['y_train']
    x_model_val = data['x_model_val']
    y_model_val = data['y_model_val']
    x_optimizer_val = data['x_optimizer_val']
    y_optimizer_val = data['y_optimizer_val']
    problem_type = data['problem_type']
    configs_encoder = individual[0:2]
    configs_model = individual[2:]
    # build models
    encoder = build_encoder(x_train, x_model_val, configs_encoder)
    if(encoder!=None):
        x_train = encoder.predict(x_train)
        x_model_val = encoder.predict(x_model_val)
        x_optimizer_val = encoder.predict(x_optimizer_val)
    model = build_model(x_train, y_train, 
                        x_model_val, y_model_val, 
                        configs_model, problem_type)
    # evaluate fitness
    fitness = model.evaluate(x_optimizer_val, y_optimizer_val)
    return fitness,

def optimize(fx, data, size_population, number_generations):
    '''function to optimize a given target function and return best possible solution

    Parameters
    ----------
    fx : function
        function that evaluates the quality of the solution
    data : dict
        a dictionary containing the arguments passed into the  optimization function
    size_population : int
        size of the population in every generation
    number_generations : int
        number of generations in GA

    Returns
    -------
    log : dict
        a dictionary containing the stats over the different populations
    solution : 
        the best structure for the given data set 
    '''
    # pylint: disable=no-member
    # configure fitness, individuals and population
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox() 
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, 7)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # creating genetic operators
    toolbox.register("evaluate", fx, data=data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # initiate the population
    pop = toolbox.population(n=size_population)
    # configure hall of fame
    hof = tools.HallOfFame(1)
    # configure stats of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # run GA algorithim and return results
    _, log = algorithms.eaSimple(pop, 
                                 toolbox, 
                                 cxpb=0.5, 
                                 mutpb=0.2, 
                                 ngen=number_generations, 
                                 stats=stats, 
                                 halloffame=hof, 
                                 verbose=True)
    solution = hof.items[0]
    return log, solution

def construct_solution(individual, data):
    '''function to construct the solution found by the optimizer

    Parameters
    ----------
    individual : list 
    data : dict

    Returns
    -------
    encoder : TF model
    model : TF model

    '''
    x_train = data['x_train'] 
    y_train = data['y_train']
    x_model_val = data['x_model_val']
    y_model_val = data['y_model_val']
    problem_type = data['problem_type']
    configs_encoder = individual[0:2]
    configs_model = individual[2:]
    # build models
    encoder = build_encoder(x_train, x_model_val, configs_encoder)
    if(encoder!=None):
        x_train = encoder.predict(x_train)
        x_model_val = encoder.predict(x_model_val)
    model = build_model(x_train, y_train, 
                        x_model_val, y_model_val, 
                        configs_model, problem_type)
    return {'encoder':encoder, 'model':model}

