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
    def __init__(self, target_params, target_obj, params, wdp, agro, observations, minmax=[], fixed_params=None):
        self.fp = fixed_params
        self.tp = target_params
        self.to = target_obj
        self.mr = ModelRerunner(self.fp, params, wdp, agro)
        self.obs = observations
        self.mm = minmax
        self.sim = None
        self.params_change = None
        self.n_calls = 0
        self.loss = None
        
        self.true = self.obs.loc[self.obs.index, self.to]
        if len(self.mm):
            self.true = ((self.true - self.mm['min'])/(self.mm['max'] - self.mm['min'])).values
       
    def __call__(self, input_params, is_train=True):
        self.n_calls += 1
        par_values = {}
        if is_train:
            for k in self.tp:
                if not k.endswith('TB'):
                    par_values[k] = input_params[k]
                else:
                    temp_list = []
                    for v1, v2 in zip(np.linspace(self.tp[k][-1][0], self.tp[k][-1][1], self.tp[k][2]),
                                      [_ for _ in input_params if _.startswith(k)]):
                        temp_list.append(v1)
                        temp_list.append(input_params[v2])
                    par_values[k] = temp_list
            self.params_change = par_values
            self.sim = self.mr(self.params_change)
            self.pred = self.sim.loc[self.obs.index, self.to]
            if len(self.mm):
                self.pred = ((self.pred - self.mm['min'])/(self.mm['max'] - self.mm['min'])).values
            self.loss = mean_squared_error(self.true, self.pred)

            return self.loss
        
        else:
            self.sim = self.mr(input_params)
            self.pred = self.sim.loc[self.obs.index, self.to]
            if len(self.mm):
                self.pred = ((self.pred - self.mm['min'])/(self.mm['max'] - self.mm['min'])).values
            
            return mean_squared_error(self.true, self.pred)