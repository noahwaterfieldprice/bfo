import numpy as np
from scipy.optimize import leastsq, curve_fit


# General fitting functions


def ls_fit(function, x_data, y_data, initial_parameters, fixed_parameters=[]):
    def residuals(parameters, x, y):
        for i, p in fixed_parameters:
            parameters[i] = p
        return y - function(x, *parameters)

    fit_parameters, jacobian = leastsq(
        residuals, initial_parameters, args=(x_data, y_data))

    res_variance = (np.sum(residuals(fit_parameters, x_data, y_data) ** 2) /
                    (len(y_data) - len(initial_parameters)))
    covariance_matrix = jacobian * res_variance
    errors = []
    for i in range(len(initial_parameters)):
        try:
            errors.append(np.absolute(covariance_matrix[i][i]) ** 0.5)
        except:
            errors.append(0.00)

    for i, p in fixed_parameters:
        fit_parameters[i] = p
    return fit_parameters, np.array(errors)


def cf_fit(function, parameters, x_data, y_data, y_data_errors=None):
    # fit function and update parameters
    if y_data_errors is None:
        pfit, pcov = \
            curve_fit(function, x_data, y_data, p0=parameters)
    else:
        pfit, pcov = \
            curve_fit(function, x_data, y_data, p0=parameters,
                      sigma=y_data_errors)
    # calculate errors
    errors = []
    for i in range(len(pfit)):
        try:
            errors.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            errors.append(0.00)
    # return errors
    return pfit, np.array(errors)



# Peak fitting functions

def fit_peak(function, x_data, y_data, initial_parameters=None):
    if initial_parameters is None:
        initial_parameters = initialise_peak_parameters(x_data, y_data)
    return ls_fit(function, x_data, y_data, initial_parameters)


def initialise_peak_parameters(x, y, constant=False):
    peak_parameters = [y.max(), x[len(x) // 2], (x[-1] - x[0]) / 2]
    if constant:
        peak_parameters.append(0)
    return peak_parameters

# Peak shape functions


def gaussian(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (c ** 2))


def gaussian_constant(x, a, b, c, d):
    return gaussian(x, a, b, c) + d


def lorentzian(x, a, b, c):
    return a * (0.5 * c) ** 2 / ((x - b) ** 2 + (0.5 * c) ** 2)


def lorentzian_constant(x, a, b, c, d):
    return lorentzian(x, a, b, c) + d


def lorentzian_squared(x, a, b, c):
    return lorentzian(x, a, b, c) ** 2


def lorentzian_squared_constant(x, a, b, c, d):
    return lorentzian_squared(x, a, b, c) + d
