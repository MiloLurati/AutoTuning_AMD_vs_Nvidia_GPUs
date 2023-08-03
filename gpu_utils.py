import random

class GPU_tuning_space:
    def __init__(self, tune_params, orig_params, fitness_dict, objective='time', multi_objective_weights=None):
        self.tune_params = tune_params
        self.fitness_dict = fitness_dict
        self.objective_var = objective
        self.objective_weights = multi_objective_weights

        # For GPU tuning, there are missing configurations that
        # need to be scored poorly
        self.fail_fit = 1e10

        # set the base settings list
        self.settings = []
        for vals in list(orig_params.values()):
            self.settings.append(vals[0])

        # Determine which settings in the list are actually going to be changed
        self.indices_to_tune = []
        it = 0
        for key in list(orig_params.keys()):
            if key in list(tune_params.keys()):
                self.indices_to_tune.append(it)
            it += 1

        self.orig_nrparams = len(orig_params.keys())
        self.tunable_nrparams = len(tune_params.keys())

    def get_config_as_str(self, params):
        if len(params) == self.tunable_nrparams:
            it = 0
            for var in params:
                self.settings[self.indices_to_tune[it]] = var
                it += 1
        elif len(params) == self.orig_nrparams:
            self.settings = params
        else:
            raise Exception("Incompatible list of parameters supplied as params argument")

        str_key = ""
        for sett in self.settings:
            str_key += str(sett) + ","
        str_key = str_key[:-1]
        return str_key

    def get_runtime(self, params):
        str_key = self.get_config_as_str(params)
        if not str_key in self.fitness_dict.keys():
            return float(self.fail_fit)

        if self.objective_var == 'times':#Stochastic version:
            if self.fitness_dict[str_key]['time'] > 1e+10:# Failed compilations have no 'times' key
                return self.fitness_dict[str_key]['time']
            times = self.fitness_dict[str_key][self.objective_var]
            random_time = random.choice(times)
            return float(random_time)
        elif isinstance(self.objective_var, str):
            return float(self.fitness_dict[str_key][self.objective_var])
        elif isinstance(self.objective_var, list):
            fit = float(self.fitness_dict[str_key][self.objective_var[0]])
            if self.objective_weights is not None:
                fit *= self.objective_weights[0]
            for i in range(1, len(self.objective_var)):
                fitvar = float(self.fitness_dict[str_key][self.objective_var[i]])
                if self.objective_weights is not None:
                    fitvar *= self.objective_weights[i]
                fit += fitvar
            return fit
        else:
            raise Exception("Unknown objective var type")


def convert_gpusetting_to_bitidxs(settings, boundary_list, sspace):
    bitidxs = []
    settings = [int(x) for x in settings.split(",")]
    it = 0
    for i, var_setting in enumerate(settings):
        var_values = list(sspace.values())[i]
        if len(var_values) == 1:
            continue
        var_setting = settings[i]
        k = var_values.index(var_setting)
        bitidxs.append(boundary_list[it][0] + k)
        it += 1
    return bitidxs
