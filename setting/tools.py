import gc, sys, yaml, os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

import pcse
from pcse.models import Wofost72_PP
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import YAMLAgroManagementReader, YAMLCropDataProvider, ExcelWeatherDataProvider
from pcse.util import WOFOST72SiteDataProvider, DummySoilDataProvider

from hyperopt import tpe, hp, fmin

class ModelRerunner(object):
    def __init__(self, fixed_params, params, wdp, agro):
        self.params = params
        self.fixed = fixed_params
        self.wdp = wdp
        self.agro = agro
        
    def __call__(self, par_values):
        # Clear any existing overrides
        self.params.clear_override()
        # Appling fixed parameters
        if self.fixed:
            for parname, value in self.fixed.items():
                self.params.set_override(parname, value)
        # Set overrides for the new parameter values
        for parname, value in par_values.items():
            self.params.set_override(parname, value)
        # Run the model with given parameter values
        wofost = Wofost72_PP(self.params, self.wdp, self.agro)
        wofost.run_till_terminate()
        df = pd.DataFrame(wofost.get_output())
        df.index = pd.to_datetime(df.day)
        return df
    
    
class ObjectiveFunctionCalculator(object):
    def __init__(self, target_params, target_obj, params, wdp, agro, observations, fixed_params=None):
        self.fixed = fixed_params
        self.target_params = target_params
        self.target_obj = target_obj
        self.modelrerunner = ModelRerunner(self.fixed, params, wdp, agro)
        self.df_obs = observations
        self.df_sim = None
        self.params_change = None
        self.n_calls = 0
        self.loss = None
        
        self.df_obs = self.df_obs.loc[self.df_obs.index, self.target_obj]
       
    def __call__(self, input_params):
        self.n_calls += 1
        par_values = {}
        for key in self.target_params:
            TARGET = [x for x in input_params.keys() if x.startswith(key)]
            if len(TARGET) == 1:
                par_values[key] = input_params[key]
            else:
                temp_list = []
                for v1, v2 in zip(np.linspace(0.0, 2.0, len(TARGET)), [input_params[_] for _ in TARGET]):
                    temp_list.append(v1)
                    temp_list.append(v2)
                par_values[key] = temp_list
        self.params_change = par_values
        self.df_sim = self.modelrerunner(self.params_change)
        diffs = self.df_sim.loc[self.df_obs.index, self.target_obj]
        diffs = self.df_obs.values - diffs.values
        diffs = diffs/np.array([1000, 0.7])
        self.loss = np.mean(diffs**2) # MSE

        return self.loss