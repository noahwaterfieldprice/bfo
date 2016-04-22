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
    errors = np.absolute(covariance_matrix.diagonal()) ** 0.5

    for i, p in fixed_parameters:
        fit_parameters[i] = p
        errors[i] = 0

    return fit_parameters, errors


def cf_fit(function, parameters, x_data, y_data, y_data_errors=None):
    # fit function and update parameters
    if y_data_errors is None:
        fit_parameters, covariance_matrix = curve_fit(
            function, x_data, y_data, p0=parameters)
    else:
        fit_parameters, covariance_matrix = curve_fit(
            function, x_data, y_data, p0=parameters, sigma=y_data_errors)
    # calculate errors
    errors = np.absolute(covariance_matrix.diagonal()) ** 0.5
    return fit_parameters, errors


def initialise_peak_parameters(x, y, constant=False):
    peak_parameters = [y.max(), x[len(x) // 2], (x[-1] - x[0]) / 2]
    if constant:
        peak_parameters.append(0)
    return peak_parameters


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
