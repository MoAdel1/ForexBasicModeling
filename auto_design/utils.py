# %% code imports
from auto_design.feature_engineering.encoder import build_encoder
from auto_design.modeling.neural_nets import build_model
from auto_design.optimizer.genetic import eval_fit, optimize, construct_solution
from auto_design.pre_processing.process import split_data, standardize

# %% function definition
def design(x, y, problem_type, size_population=15, number_generations=10):
    '''main function that utilize the rest of the submodules to build a fully functional ML pipeline automatically

    Parameters
    ----------
    x : numpy ndarray
        the trainig features
    y : numpy ndarray
        the training targets
    problem_type : str
        the type pf the problem to be solved
    size_population : int
        size of the population in every generation
    number_generations : int
        number of generations in GA

    Returns
    -------
    pipeline : dict
        a python dictionary containing the scaler, encoder and model
    '''
    # standardize data and return scaler
    x, scaler = standardize(x)
    # split the data
    x_train, y_train, x_model_val, y_model_val, x_optimizer_val, y_optimizer_val = split_data(x, y, 0.3, 0.1)
    # run optimizer and return best feature reduction stage and model
    data = {'x_train':x_train, 
            'y_train':y_train, 
            'x_model_val':x_model_val, 
            'y_model_val':y_model_val, 
            'x_optimizer_val':x_optimizer_val, 
            'y_optimizer_val':y_optimizer_val,
            'problem_type':problem_type}
    log, solution = optimize(eval_fit, data=data, size_population=size_population, number_generations=number_generations)
    # construct solution
    pipeline = construct_solution(solution, data)
    pipeline['scaler'] = scaler
    # function return
    return(pipeline, log)
    