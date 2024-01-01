
import numpy as np


# define the FI curve function. for either excitatory neurons and inhibitory neurons.
def current_to_frequency(input_current,population_type,p):
    # input_current is all the current to target neuron.
    # population_type is either 'E' for excitatory neurons or 'I' for inhibitory neurons.
    # p is a dict defining all parameters
    # NOTE: using python broadcasting for ones subtraction, so should work for multi-area case too
    if population_type == 'E':
        a = p['a_E']
        b = p['b_E']
        d = p['d_E']
        r = np.divide((a*input_current - b),(1 - np.exp(-d*(a*input_current - b))))
    if population_type == 'I':
        c_I = p['c_I']
        g_2 = p['g_2']
        r_0 = p['r0_I']
        c_0 = p['c_0']
        r = np.maximum((1/g_2)*(c_I*input_current - c_0) + r_0,0)
    if population_type == 'th':
        a = p['a_E']
        b = p['b_E']
        d = p['d_E']
        r = np.divide((a * input_current - b), (1 - np.exp(-d * (a * input_current - b))))

    return r

# define the right hand side of NMDA equation.
def NMDA_deriv(S_NMDA_prev, rate_now, p):
    # S_NMDA_prev is S_NDMA at the previous time step.
    # rate_now is firing rate at the current time step. (actually doesn't matter much)
    # p is dict of all parameters

    return -S_NMDA_prev / p['tau_NMDA'] + p['gamma_NMDA'] * (1 - S_NMDA_prev) * rate_now


def GABA_deriv(S_GABA_prev, rate_now, parameters):
    return -S_GABA_prev / parameters['tau_GABA'] + parameters['gamma_GABA'] * rate_now

def AMPA_th_deriv(S_AMPA_prev, rate_now, parameters):
    return -S_AMPA_prev / parameters['tau_AMPA'] + parameters['gamma_AMPA'] * rate_now

