#%% code imports
import os
import json


#%% main code
with open(os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs.json"))) as json_file:
    configs = json.load(json_file)
if(configs['local']==True):
    with open(os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs_local.json"))) as json_file:
        configs = json.load(json_file)