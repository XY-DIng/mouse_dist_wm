
# let's get what we need together
# using python 3
import numpy as np
import numpy.matlib
import scipy as sp
from scipy import stats
import pandas

import brian2
from adjustText import adjust_text
import pickle
import os
import matplotlib.pyplot as plt
import json

from modelHelper import current_to_frequency, NMDA_deriv, GABA_deriv, AMPA_th_deriv

class model():
    def __init__(self, parameters, disconnected_network, conn_cxcx, pref_matrix, hierarchy_df, normPVgrad_df, normSSTgrad_df, area_list, thcxmodel, conn_thcx, conn_cxth, thal_areas_list):
        """

        :param parameters:
        :param disconnected_network:
        :param conn_cxcx:
        :param pref_matrix:
        :param hierarchy_df:
        :param normPVgrad_df:
        :param normSSTgrad_df:
        :param area_list:
        :param thcxmodel:
        :param conn_thcx:
        :param conn_cxth:
        :param thal_areas_list:
        """
        self.num_areas = conn_cxcx.shape[0]
        self.n_areas = self.num_areas
        self.hierarchy_df = hierarchy_df
        self.normPVgrad_df = normPVgrad_df
        self.normSSTgrad_df = normSSTgrad_df
        self.area_list = area_list
        self.pref_matrix = pref_matrix
        self.parameters = parameters
        self.thcxmodel = thcxmodel
        self.conn_cxth = conn_cxth
        self.conn_thcx = conn_thcx
        self.thal_areas_list = thal_areas_list
        ######## LOCAL ########
        # self.J = np.array([[0, parameters['g_E_cross'], 0],
        #                    [parameters['g_E_cross'], 0, 0],
        #                    [parameters['g_E_I'], parameters['g_E_I'], parameters['g_Iself']]]).T * brian2.amp
        # what is the role of J_NMDA? g_Iself is larger than 0, so it will be in J_NMDA. Is this correct?  J_NMDA is only used for
        # the cross pop ex connections, so the third row is not used. For IN connecitons I implement it sepereately.

        self.J = np.array([[0, parameters['g_E_cross'], 0],
                           [parameters['g_E_cross'], 0, 0],
                           [0, 0, 0]]).T * brian2.amp

        self.pops = ['E1', 'E2', 'I']
        self.pops_column_list = ['from ' + mystring for mystring in self.pops]
        self.pops_row_list = ['to ' + mystring for mystring in self.pops]

        self.df_J = pandas.DataFrame(self.J, columns=self.pops_column_list, index=self.pops_row_list)

        self.J_NMDA = self.J * ((self.J > 0).astype(np.int))
        # self.J_GABA = self.J*((self.J<0).astype(np.int))#J_GABA is not useful here, local I input are replaced by J_II_gradient


        self.local_EE_gradient = parameters['J_NS_grad_min'] * np.ones((self.n_areas, 1))
        # self.local_EE_gradient = parameters['J_NS_grad_min'] + parameters['JEE_scaling_factor']*self.hierarchy

        self.local_EI_gradient = parameters['J_G_EI_min'] + parameters['JEI_scaling_factor'] * (np.array(normPVgrad_df))

        # self.LR_EE_gradient = self.local_EE_gradient/np.max(self.local_EE_gradient) # original setting from JM's code
        # self.LR_EE_gradient = 1  # set how local gradient affect LR EE connections, 1 means no effect

        # Scale the local E to I,local I to I, LR EI synapses to lie along the hierarachy
        # self.local_IE_gradient = 0.5*(parameters['J_plus']-self.local_EE_gradient*parameters['g_E_self']-parameters['g_E_cross'])/(parameters['zeta']*brian2.nA)
        # parameters['g_I_E'] = 20*np.max(self.local_IE_gradient) *brian2.nA      # nA also a key param need to tune
        #                         # larger g_I_E could give us larger PV firing rate.

        self.local_IE_gradient = parameters['J_N_IE_min'] + parameters['JIE_scaling_factor'] * (np.array(normPVgrad_df))
        # self.local_IE_gradient = parameters['J_NS_grad_min'] * np.ones((self.n_areas, 1))  # renormalize localIE_gradient to 1 so that the formula is consistent.
        self.local_II_gradient = parameters['J_G_II_min'] + parameters['JII_scaling_factor'] * (np.array(normPVgrad_df))

        # self.LR_IE_gradient    = self.local_IE_gradient/np.max(self.local_IE_gradient) # original setting from JM's code
        # self.LR_IE_gradient = 1  # set how local gradient affect LR IE connections, 1 means no effect

        ####### LONG-RANGE CONNECTIONS ########

        # Compress FLNf
        self.disconnected_network = disconnected_network
        if disconnected_network == False:
            W_squish = np.power(conn_cxcx, parameters['kSquishFln'])
            if self.parameters['WNormalizedbyRow']:
                #     fln_rowtotal = np.sum(fln_squish, axis=1) # sum then normalize across the rows
                # #     fln_rowtotal = np.sum(fln_squish,axis=0) # sum then normalize across the columns
                #     fln_rowtotal_mat = np.matlib.repmat(fln_rowtotal, self.num_areas,1).T
                #     fln_squishnorm = parameters['kFln']*fln_squish/fln_rowtotal_mat
                W_rowtotal = np.sum(W_squish, axis=1) # sum the rows
                self.W_rowtotal_mat = np.matlib.repmat(W_rowtotal, self.num_areas, 1).T
                W_squishnorm = W_squish/self.W_rowtotal_mat
                W_squishnorm = W_squishnorm/np.max(W_squishnorm)

            else:
                # use the max value of conn_cxcx to normalize.
                W_max = np.max(W_squish)
                W_min = np.min(W_squish)
                W_squishnorm = (W_squish)/(W_max)
            # Make feedforward connections
            self.W_E = W_squishnorm * (pref_matrix)
            # self.W_E = self.W_squishnorm
            # Make feedback connections target more inhibitory cells
            self.W_I = W_squishnorm * (1-pref_matrix)
            np.fill_diagonal(self.W_E, 0)
            np.fill_diagonal(self.W_I, 0)
        elif disconnected_network == True:
            self.W_E = np.zeros(np.shape(conn_cxcx))
            self.W_I = np.zeros(np.shape(conn_cxcx))

        self.LR_pop_conn_mat = np.array(
            [[parameters['LR_E_self'], parameters['LR_E_cross'], parameters['LR_I_E']],
             [parameters['LR_E_cross'], parameters['LR_E_self'], parameters['LR_I_E']]]).T

        self.num_pops = self.LR_pop_conn_mat.shape[0]
        self.num_E_pops = 2

        # Choose initial values for rates and synapse variables
        self.R0vec = np.array([parameters['r0_E'], parameters['r0_E'], parameters['r0_I']])# TODO changes happen
        self.R0 = np.matlib.repmat(self.R0vec, self.num_areas, 1) * brian2.Hz
        self.S_NMDA0 = 0 * np.concatenate((np.ones((self.num_areas, 2)), np.zeros((self.num_areas, self.num_pops - 2))),
                                          axis=1)
        self.S_GABA0 = 0 * np.concatenate((np.zeros((self.num_areas, 2)), np.ones((self.num_areas, self.num_pops - 2))),
                                          axis=1)

        self.df_LR_pop_conn = pandas.DataFrame(self.LR_pop_conn_mat, columns=self.pops_column_list[:2],
                                               index=self.pops_row_list)

        # print(np.concatenate((np.matlib.repmat(local_EE_gradient,1,2),local_IE_gradient),axis=1))

        # # Set up simulation parameters & system initialization
        # def __init__(self,var):

        self.dt = parameters['dt']
        self.trial_length = parameters['trial_length']

        self.num_iterations = int(self.trial_length / self.dt)
        self.time = np.arange(0, self.trial_length, self.dt)

        local_optoinh_search = False

        # inh_area = None
        # if inh_area !=None:
        #     inh_area_idx = area_list.index(inh_area)

        self.inh_multiarea = parameters['inh_multiarea']

        if self.inh_multiarea != None:
            self.len_inhmulti = len(self.inh_multiarea)
            self.inh_multiarea_idx = np.zeros(self.len_inhmulti)
            for j in np.arange(0, self.len_inhmulti):
                self.inh_multiarea_idx[j] = area_list.index(self.inh_multiarea[j])
        # inhibition_duration = 5 * brian2.second

        if self.thcxmodel ==True:
            self.num_th_areas = self.conn_thcx.shape[1] # TODO not sure if this should be 0 or 1
            self.num_th_pops = 2
            self.stim_strength_th = parameters['stim_strength_th']
            conn_thcx_squish = np.power(conn_thcx, parameters['kSquishFln'])
            conn_cxth_squish = np.power(conn_cxth, parameters['kSquishFln'])
            # conn_thcx_squish = np.power(conn_thcx, 1)
            # conn_cxth_squish = np.power(conn_cxth, 1)
            self.W_th_cx = conn_thcx_squish/np.max(conn_thcx_squish)
            self.W_cx_th = conn_cxth_squish/np.max(conn_cxth_squish)

            self.R_th0 = np.zeros((self.num_th_areas, self.num_th_pops)) * brian2.Hz
            self.S_AMPA_th0 = np.zeros((self.num_th_areas, self.num_th_pops))
            self.D_th_0 = np.zeros((self.num_th_areas, self.num_th_pops))

            self.R_th = np.zeros((self.num_iterations, self.num_th_areas, self.num_th_pops)) *brian2.Hz
            self.R_th[0,:,:] = self.R_th0
            self.S_AMPA_th = np.zeros((self.num_iterations, self.num_th_areas, self.num_th_pops))
            self.S_AMPA_th[0,:,:] = self.S_AMPA_th0
            self.D_th = np.zeros((self.num_iterations, self.num_th_areas, self.num_th_pops)) # D is the STD factor
            self.D_th[0, :, :] = self.D_th_0

            self.I_ext_th = np.zeros((self.num_iterations, self.num_th_areas, self.num_th_pops)) * brian2.amp
            self.I_0_th = np.zeros((self.num_th_areas, self.num_th_pops)) * brian2.pA # background th input
            self.I_0_th[:, :] = parameters['I0_th_E']
            self.I_noise_th = np.zeros((self.num_th_areas, self.num_th_pops)) * brian2.pA
            self.I_th_total = np.zeros((self.num_iterations, self.num_th_areas, self.num_th_pops)) * brian2.pA

            self.I_th_cx = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA# current from th to cx, size should match cortical areas.
            self.I_cx_th = np.zeros((self.num_iterations, self.num_th_areas, self.num_th_pops)) * brian2.pA# current from cx to th


        # Preassign rate and synapse matrices
        self.R = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.Hz
        self.R[0, :, :] = self.R0
        self.S_NMDA = np.zeros((self.num_iterations, self.num_areas, self.num_pops))
        self.S_NMDA[0, :, :] = self.S_NMDA0
        self.S_GABA = np.zeros((self.num_iterations, self.num_areas, self.num_pops))
        self.S_GABA[0, :, :] = self.S_GABA0

        # # Preassign external inputs
        self.I_ext = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.amp
        self.stim_on = parameters['stim_on']  # s
        self.stim_off = parameters['stim_off']  # s

        self.stim_strength = parameters['stim_strength']
        self.BLA_stim_strength = parameters['BLA_stim_strength']
        self.distractor_strength = parameters['distractor_strength']
        self.inhibition_strength = parameters['inhibition_strength']

        # inhibition_duration = self.trial_length - stim_off
        self.inhibition_duration = 5 * brian2.second
        self.inhibition_delay = 7 * brian2.second
        self.inhibition_on = self.stim_off + self.inhibition_delay  # s
        self.inhibition_off = self.stim_off + self.inhibition_delay + self.inhibition_duration  # s
        # inhibition_off = self.trial_length #s

        # second stimulus to system
        self.second_stimulus_delay = 5 * brian2.second
        self.stim2_on = self.stim_off + self.second_stimulus_delay
        self.stim2_off = self.stim2_on + self.stim_off - self.stim_on

        self.noiseStarttime = 0.5 * brian2.second
        self.noiseEndtime = self.trial_length
        # extnoisemean = 0.05*brian2.nA
        # extnoisestd = 0.05*brian2.nA
        self.extnoisemean = 0.00 * brian2.nA
        self.extnoisestd = 0.005 * brian2.nA

        # Create matrices in which we can store the currents
        self.I_LR_NMDA = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        self.I_local_cross_NMDA = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        self.I_local_NMDA = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        self.I_local_GABA = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        # self.I_local_GABA =  np.zeros((self.num_iterations,self.num_areas,self.num_pops)) * brian2.pA
        self.I_soma_dend = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        self.I_total = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        self.I_exc_dend = np.zeros((self.num_iterations, self.num_areas, self.num_E_pops)) * brian2.pA
        self.I_inh_dend = np.zeros((self.num_iterations, self.num_areas, self.num_E_pops)) * brian2.pA

        # # Define background inputs
        self.I_0 = np.zeros((self.num_areas, self.num_pops)) * brian2.pA

        self.I_0[:, :] = np.matlib.repmat(parameters['I0_E_mean'],1,3)

        # self.I_0[:, [self.pops.index('E1'), self.pops.index('E2')]] = np.matlib.repmat(parameters['I0_E_min'] + \
        #                                                                                np.array(
        #                                                                                    self.hierarchy_df) * (
        #                                                                                    parameters['I0_E_max'] -
        #                                                                                    parameters['I0_E_max']), 1,
        #                                                                                2)

        self.I_0[:, 2] = parameters['I0_I']

        if thcxmodel:
            self.g_th_cx_E = np.zeros(self.num_areas) * brian2.nA
            if parameters['g_th_cx_E_mode'] == 'linear':
                self.g_th_cx_E = parameters['g_th_cx_E_min'] + (1 - np.array(self.normPVgrad_df).flatten()) * (parameters['g_th_cx_E_max'] - parameters['g_th_cx_E_min'])
            elif parameters['g_th_cx_E_mode'] == 'logistic':
                self.g_th_cx_E = parameters['g_th_cx_E_min'] + (parameters['g_th_cx_E_max'] - parameters['g_th_cx_E_min']) * 1/(1 + np.exp(-parameters['g_th_cx_E_logistic_k'] * (parameters['g_th_cx_E_midpoint'] - np.array(self.normPVgrad_df).flatten()) ))
                # self.g_th_cx_E = parameters['g_th_cx_E_min'] + (parameters['g_th_cx_E_max'] - parameters[
                #     'g_th_cx_E_min']) * 1 / (1 + np.exp(-parameters['g_th_cx_E_logistic_k'] * (
                # np.array(self.hierarchy_df).flatten() - parameters['g_th_cx_E_midpoint'])))
            else: print('Undefined mode')


            self.g_th_cx_I = np.zeros(self.num_areas) * brian2.nA

            if parameters['g_th_cx_I_mode'] == 'linear':
                self.g_th_cx_I = parameters['g_th_cx_I_min'] + (np.array(self.normPVgrad_df).flatten()) * (parameters['g_th_cx_I_max'] - parameters['g_th_cx_I_min'])
            elif parameters['g_th_cx_I_mode'] == 'logistic':
                self.g_th_cx_I = parameters['g_th_cx_I_min'] + (parameters['g_th_cx_I_max']- parameters['g_th_cx_I_min']) * 1/(1 + np.exp(- parameters['g_th_cx_I_logistic_k'] * (np.array(self.normPVgrad_df).flatten() - parameters['g_th_cx_I_midpoint']) ))
                # self.g_th_cx_I = parameters['g_th_cx_I_min'] + (parameters['g_th_cx_I_max'] - parameters[
                # 'g_th_cx_I_min']) * 1 / (1 + np.exp(
                # -self.g_th_cx_I_logistic_k * (1- np.array(self.hierarchy_df).flatten() - parameters['g_th_cx_I_midpoint'])))

        # Let's set up the noise. We will model the noise as an Ornstein-Uhlenbeck process.

        # Gaussian noise. mean 0, std 1. Dims: timesteps, local populations, areas
        self.eta = np.random.normal(loc=0.0, scale=1.0, size=(self.num_iterations, self.num_areas, self.num_pops))

        # prepare the right hand side of the above equation
        self.noise_rhs = self.eta * (
        (np.sqrt(parameters['tau_noise'] * np.power(parameters['std_noise'], 2)) * np.sqrt(self.dt)) / parameters[
            'tau_noise'])  # effectively std*sqrt(dt/tau)

        # self.noise_rhs[:, :, 2] = 0  # Jorge only put noise on the E pops, I don't think this is correct
        # if noise_network ==True:
        #     noise_rhs[:,area_list.index('SSp-bfd'),pops.index('E1')] = extnoisestd/parameters['std_noise']*noise_rhs[:,area_list.index('SSp-bfd'),pops.index('E1')]

        self.I_noise = np.zeros((self.num_areas, self.num_pops)) * brian2.pA

        #     print(self.J_NMDA)
        # print(J_GABA)#J_GABA is not useful here, local I input are replaced by J_II_gradient

    def remove_I0E(self, arealist):
        for area in arealist:
            idxk = self.area_list.index(area)
            self.I_0[idxk, [0, 1]] = 0

    def cut_connection_thcx(self, areapairlist,printout=True):
        for (areai,areaj) in areapairlist:
            idxi = self.thal_areas_list.index(areai)
            idxj = self.area_list.index(areaj)
            self.W_th_cx[idxj,idxi] = 0
            if printout ==True:
                print((areai,areaj))
            # self.W_cx_th[idxi,idxj] = 0
        if printout==True:
            print(self.W_th_cx)

    def add_input(self, stim_strength, inputareas, stim_on, stim_off, stim_type, printout=True):
        self.inputareas = inputareas
        self.stim_strength = stim_strength
        if self.disconnected_network == True:
            self.I_ext[int(stim_on / self.dt):int(stim_off / self.dt), :,
            0] = stim_strength
            #     elif noise_network ==True:
            #         noiseSeq = np.random.normal(extnoisemean,extnoisestd,size=int(noiseEndtime/dt)-int(noiseStarttime/dt))*brian2.amp

            #     #Toggle: noise is given to VISp. or to all areas

            #     #     noiseSeq = noiseSeq.reshape((-1,1))
            #     #     I_ext[int(noiseStarttime/dt):int(noiseEndtime/dt),:,pops.index('E1')] = noiseSeq
            #     #     I_ext[int(noiseStarttime/dt):int(noiseEndtime/dt),:,pops.index('E2')] = noiseSeq
            #         I_ext[int(noiseStarttime/dt):int(noiseEndtime/dt),area_list.index('VISp'),pops.index('E1')] = noiseSeq
            #         I_ext[int(noiseStarttime/dt):int(noiseEndtime/dt),area_list.index('VISp'),pops.index('E2')] = noiseSeq
            #     elif BLA_simulation == True:
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ACAd'),pops.index('E1')] = 0.38*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ACAv'),pops.index('E1')] = 0.21*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('AId'),pops.index('E1')] = 0.35*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('AIv'),pops.index('E1')] = 0.31*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ILA'),pops.index('E1')] = 0.49*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('MOp'),pops.index('E1')] = 0.29*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('MOs'),pops.index('E1')] = 0.76*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('PL'),pops.index('E1')] = 0.65*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ORBl'),pops.index('E1')] = 0.17*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ORBm'),pops.index('E1')] = 0.28*BLA_stim_strength
            #         I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ORBvl'),pops.index('E1')] = 0.20*BLA_stim_strength
        else:
            # Let's apply external stimulation to V1 populations E1 & E2  # STIMPATTERN
            # change V1 to VISp
            for area in inputareas:
                if stim_type=='E1':
                    self.I_ext[int(stim_on / self.dt):int(stim_off / self.dt), self.area_list.index(area),
                       0] = stim_strength
                elif stim_type == 'E2':
                    self.I_ext[int(stim_on / self.dt):int(stim_off / self.dt), self.area_list.index(area),
                    1] = stim_strength
                elif stim_type =='both':
                    self.I_ext[int(stim_on / self.dt):int(stim_off / self.dt), self.area_list.index(area),
                    [0, 1]] = stim_strength
                elif stim_type == 'I':
                    self.I_ext[int(stim_on / self.dt):int(stim_off / self.dt), self.area_list.index(area),
                    2] = stim_strength
                if printout:
                    print(area)
                # some ref parameters when stimulating VISp 0.3-0.32: State 1; 0.33-0.34: State 2   0.35-0.37: state 3   0.39-: State 4
                # add second stimulus to show the switch of the attractor state
                #     I_ext[int(stim2_on/dt):int(stim2_off/dt),area_list.index('PL'),pops.index('E1')] = stim_strength
                #     I_ext[int(distract_on/dt):int(distract_off/dt),area_list.index('VISp'),pops.index('E2')] = distractor_strength

                # simulate the effect of an transient input from MD.
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ACAd'),self.pops.index('E1')] = 0.147*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('AId'),self.pops.index('E1')] = 0.291*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('AIv'),self.pops.index('E1')] = 0.333*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ILA'),self.pops.index('E1')] = 0.481*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('MOs'),self.pops.index('E1')] = 0.211*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ORBl'),self.pops.index('E1')] = 0.443*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ORBm'),self.pops.index('E1')] = 0.477*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ORBvl'),self.pops.index('E1')] = 0.400*stim_strength
                #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('PL'),self.pops.index('E1')] = 0.637*stim_strength

                # when looking for multiple attractors, we stimulate different areas. the code is in next block.
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('VISpm'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('SSp-bfd'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('AUDp'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ACAv'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ACAd'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('PL'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ILA'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('RSPv'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('RSPd'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ORBl'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ORBm'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('ORBl'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('RSPagl'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('AIp'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('AId'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('AIv'),pops.index('E1')] = stim_strength
                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('VISa'),pops.index('E1')] = stim_strength



                #     I_ext[int(stim_on/dt):int(stim_off/dt),area_list.index('AUDp'),pops.index('E2')] = 1.25*stim_strength
                # I_ext[int(distract_on/dt):int(distract_off/dt),area_list.index('SSp-bfd'),pops.index('E2')] = distractor_strength
                # testing the effect of area lesion
                #     I_ext[int(stim_on/dt):,area_list.index('AIv'),[pops.index('E1'),pops.index('E2')]] = inhibition_strength
                #     if inh_area !=None:
                #         I_ext[int(inhibition_on/dt):int(inhibition_off/dt),inh_area_idx,[pops.index('E1'),pops.index('E2')]] = inhibition_strength
            if self.inh_multiarea != None:
                for j in np.arange(0, self.len_inhmulti):
                    self.I_ext[int(self.inhibition_on / self.dt):int(self.inhibition_off / self.dt),
                    int(self.inh_multiarea_idx[j]),
                    [0, 1]] = self.inhibition_strength
                    #             I_ext[int((inhibition_off+5*brian2.second)/dt):int((inhibition_off+6*brian2.second)/dt),int(inh_multiarea_idx[j]),pops.index('E1')] = stim_strength

    # if local_optoinh_search ==True:
    #         # need to change inh_area_idx to multiarea format.
    #         self.I_ext[int(self.inhibition_on/self.dt):int(self.inhibition_off/self.dt),inh_area_idx,[self.pops.index('E1'),self.pops.index('E2')]] = self.inhibition_strength

    def add_input_TH(self, stim_strength, inputareas, stim_on, stim_off, stim_type):
        self.inputareas = inputareas
        self.stim_strength = stim_strength

        for area in inputareas:
            if stim_type=='E1':
                self.I_ext_th[int(stim_on / self.dt):int(stim_off / self.dt), self.thal_areas_list.index(area),
                   0] = stim_strength
            elif stim_type == 'E2':
                self.I_ext_th[int(stim_on / self.dt):int(stim_off / self.dt), self.thal_areas_list.index(area),
                1] = stim_strength
            elif stim_type =='both':
                self.I_ext_th[int(stim_on / self.dt):int(stim_off / self.dt), self.thal_areas_list.index(area),
                [0,1]] = stim_strength
            print(area)

    def add_input_TH_new(self, stim_strength, inputareas, stim_on, stim_off, stim_type,rampingperiod = 1/5):
        self.inputareas = inputareas
        self.stim_strength = stim_strength
        dt = self.dt
        # stim_0 = int(stim_on / self.dt)
        # stim_last = int(stim_off / self.dt)
        # stim_1 = int(((stim_off-stim_on)/10 + stim_on) / self.dt)
        # stim_2 = int(((stim_off - stim_on) / 10*2 + stim_on) / self.dt)
        # stim_3 = int(((stim_off - stim_on) / 10*3 + stim_on) / self.dt)
        # for area in inputareas:
        #     if stim_type == 'E1':
        #         # self.I_ext_th[stim_0:stim_last, self.thal_areas_list.index(area),
        #         # [0, 1]] = stim_strength
        #         #
        #         self.I_ext_th[stim_0:stim_1, self.thal_areas_list.index(area),
        #         0] = stim_strength / 4
        #         self.I_ext_th[stim_1:stim_2, self.thal_areas_list.index(area),
        #         0] = stim_strength / 4 * 2
        #         self.I_ext_th[stim_2:stim_3, self.thal_areas_list.index(area),
        #         0] = stim_strength / 4 * 3
        #         self.I_ext_th[stim_3:stim_last, self.thal_areas_list.index(area),
        #         0] = stim_strength
        #
        stim_0 = int(stim_on/dt)
        stim_1 = int( ( (stim_off-stim_on)*rampingperiod + stim_on)/dt)
        stim_last = int(stim_off/dt)
        for area in inputareas:
            if stim_type == 'E1':

                for k in np.arange(stim_0,stim_1):
                    self.I_ext_th[k, self.thal_areas_list.index(area),
                    0] = stim_strength * (k-stim_0)/(stim_1-stim_0)
                self.I_ext_th[stim_1:stim_last, self.thal_areas_list.index(area),
                    0] = stim_strength


                print(area)
            else: print('unknown type')
        return

    def add_input_TH_smooth(self, stim_strength, inputareas, stim_on, stim_off, stim_type,rampingperiod = 1/5):
        self.inputareas = inputareas
        self.stim_strength = stim_strength
        dt = self.dt

        stim_0 = int(stim_on/dt)
        stim_1 = int( ( (stim_off-stim_on)*rampingperiod + stim_on)/dt)
        stim_last = int(stim_off/dt)
        for area in inputareas:
            if stim_type == 'E1':

                for k in np.arange(stim_0,stim_1):
                    self.I_ext_th[k, self.thal_areas_list.index(area),
                    0] = stim_strength * (k-stim_0)/(stim_1-stim_0)
                self.I_ext_th[stim_1:stim_last, self.thal_areas_list.index(area),
                    0] = stim_strength


                print(area)
            else: print('unknown type')
        return



    def add_subcorx_input_old(self, subcorx_stim_list, subcorx_inputareas):
        self.subcorx_inputareas = subcorx_inputareas
        self.subcorx_stim_list = subcorx_stim_list

    # STIMPATTERN adding input to areas.
        for stim, area in zip(subcorx_stim_list,subcorx_inputareas):
            print(area)
            print(stim)
            #     I_ext[int(stim2_on/dt):int(stim2_off/dt),area_list.index('PL'),pops.index('E1')] = stim_strength
            #     I_ext[int(distract_on/dt):int(distract_off/dt),area_list.index('VISp'),pops.index('E2')] = distractor_strength

            # simulate the effect of an transient input from MD.
            self.I_ext[int(self.stim2_on / self.dt):int(self.stim2_off / self.dt), self.area_list.index(area),0] = stim
            # self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ACAd'),self.pops.index('E1')] = stim
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('AId'),self.pops.index('E1')] = 0.291*stim_strength
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('AIv'),self.pops.index('E1')] = 0.333*stim_strength
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ILA'),self.pops.index('E1')] = 0.481*stim_strength
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('MOs'),self.pops.index('E1')] = 0.211*stim_strength
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ORBl'),self.pops.index('E1')] = 0.443*stim_strength
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ORBm'),self.pops.index('E1')] = 0.477*stim_strength
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('ORBvl'),self.pops.index('E1')] = 0.400*stim_strength
            #         self.I_ext[int(self.stim2_on/self.dt):int(self.stim2_off/self.dt),self.area_list.index('PL'),self.pops.index('E1')] = 0.637*stim_strength


    def run_sim(self,vocal_prompt =False):
        print('simulation start: ')
        parameters = self.parameters
        # run simulaition
        for i_t in range(1, self.num_iterations):
            # update noise current
            self.I_noise = self.I_noise + -self.I_noise * (self.dt / parameters['tau_noise']) + self.noise_rhs[i_t - 1,
                                                                                                :, :]

            #  Long range input to E-pops is a sum of NMDA only (see large_scale_monkey_interne
            # urons for version with AMPA)
            #  weighted by long-range connectivity strength(FLN), Cross & Self population weights
            # Long range input(I_LR) are mediated through NMDA current.

            # Calculate long-range NMDA inputs
            # Note, we're using python broadcasting to implement the gradient. If replicating in Matlab, will need to use repmat

            # Long range to E populations
            self.I_LR_NMDA[i_t - 1, :, :2] = parameters['mu_EE'] * (
            self.W_E.dot(self.S_NMDA[i_t - 1, :, :2])).dot(self.LR_pop_conn_mat[:2, :].T) #  add * self.LR_EE     _gradient if local gradient affect LR gradient.
            # note that LR_EE gradint =1
            # Long range to I population
            self.I_LR_NMDA[i_t - 1, :, 2] = parameters['mu_IE']  * (
            self.W_I.dot(self.S_NMDA[i_t - 1, :, :2])).dot(self.LR_pop_conn_mat[2, :].T)  # add # np.squeeze(self.LR_IE_gradient) if local gradient affect LR gradient.
            # note that LR_IE gradint =1
            # local non-graded NMDA
            self.I_local_cross_NMDA[i_t - 1, :, :] = self.J_NMDA.dot(self.S_NMDA[i_t - 1, :, :].T).T

            # Note, we're using python broadcasting to implement the gradient. If replicating in Matlab, will need to use repmat
            # Note that Jorge only applies the hierarchy to self connections, hence we need to do this line-by-line
            self.I_local_NMDA[i_t - 1, :, 0] = np.squeeze(self.local_EE_gradient) * parameters[
                'g_E_self'] * self.S_NMDA[i_t - 1, :, 0]
            self.I_local_NMDA[i_t - 1, :, 1] = np.squeeze(self.local_EE_gradient) * parameters[
                'g_E_self'] * self.S_NMDA[i_t - 1, :, 1]

            # sum up all the local NMDA current onto I cells
            self.I_local_NMDA[i_t - 1, :, 2] = np.squeeze(self.local_IE_gradient) * parameters[
                'g_I_E'] * self.S_NMDA[i_t - 1, :, 0] \
                                                    + np.squeeze(self.local_IE_gradient) * parameters[
                'g_I_E'] * self.S_NMDA[i_t - 1, :, 1]

            self.I_local_GABA[i_t - 1, :, 0] = np.squeeze(self.local_EI_gradient) * parameters[
                'g_E_I'] * self.S_GABA[i_t - 1, :, 2]
            self.I_local_GABA[i_t - 1, :, 1] = np.squeeze(self.local_EI_gradient) * parameters[
                'g_E_I'] * self.S_GABA[i_t - 1, :, 2]
            self.I_local_GABA[i_t - 1, :, 2] = np.squeeze(self.local_II_gradient) * parameters[
                'g_Iself'] * self.S_GABA[i_t - 1, :, 2]
            # sum up all the local GABA current onto E and I cells
            # I_local_GABA[i_t-1,:,:] = J_GABA.dot(S_GABA[i_t-1,:,:].T).T
            # J_GABA is not useful here, local I input are replaced by J_II_gradient

            # Define total input current as sum of local NMDA & GABA inputs, with background and external currents,
            # noise and long-range NMDA inputs
            if self.thcxmodel == False:
                self.I_total[i_t - 1, :, :] = self.I_local_NMDA[i_t - 1, :, :] + self.I_local_cross_NMDA[i_t - 1, :,
                                                                                 :] \
                                              - self.I_local_GABA[i_t - 1, :, :] + self.I_0 + self.I_ext[i_t - 1, :,
                                                                                              :] + self.I_noise + self.I_LR_NMDA[
                                                                                                                  i_t - 1,
                                                                                                                  :, :]
            else:
                # define th to cx current and new I total for cx areas.
                # self.I_th_cx[i_t - 1, :, 0] = parameters['g_th_cx_E']*np.dot(self.W_th_cx,  self.D_th[i_t-1,:,0] * self.S_AMPA_th[i_t-1,:,0]) # D will change all TH-CX current
                # self.I_th_cx[i_t - 1, :, 1] = parameters['g_th_cx_E'] * np.dot(self.W_th_cx,  self.D_th[i_t-1,:,1] * self.S_AMPA_th[i_t - 1, :, 1])
                # self.I_th_cx[i_t - 1, :, 2] = parameters['g_th_cx_I']*np.dot(self.W_th_cx, (self.D_th[i_t-1,:,0] * self.S_AMPA_th[i_t-1,:,0]
                #                                                                              + self.D_th[i_t-1,:,1] * self.S_AMPA_th[i_t-1,:,1]))

                # next two lines use constant gthcxE.
                # self.I_th_cx[i_t - 1, :, 0] = parameters['g_th_cx_E'] * np.dot(self.W_th_cx, self.S_AMPA_th[i_t - 1, :,0])  # D will change all TH-CX current
                # self.I_th_cx[i_t - 1, :, 1] = parameters['g_th_cx_E'] * np.dot(self.W_th_cx, self.S_AMPA_th[i_t - 1, :, 1])

                self.I_th_cx[i_t - 1, :, 0] = self.g_th_cx_E * np.dot(self.W_th_cx, self.S_AMPA_th[i_t - 1, :,0])
                self.I_th_cx[i_t - 1, :, 1] = self.g_th_cx_E * np.dot(self.W_th_cx, self.S_AMPA_th[i_t - 1, :, 1])
                self.I_th_cx[i_t - 1, :, 2] = self.g_th_cx_I * np.dot(self.W_th_cx, (self.S_AMPA_th[i_t - 1, :, 0] + self.S_AMPA_th[i_t - 1, :, 1]))

                self.I_total[i_t - 1, :, :] = self.I_local_NMDA[i_t - 1, :, :] + self.I_local_cross_NMDA[i_t - 1, :, :] \
                                          - self.I_local_GABA[i_t - 1, :, :] + self.I_0 + self.I_ext[i_t - 1, :, :] \
                                              + self.I_noise + self.I_LR_NMDA[i_t - 1,:, :] \
                                              + self.I_th_cx[i_t-1,:,:]

            if self.thcxmodel == True:
                # define cx to th current and I total for th areas
                self.I_cx_th[i_t-1, :, 0] = parameters['g_cx_th']*np.dot(self.W_cx_th,self.S_NMDA[i_t-1,:,0])
                self.I_cx_th[i_t - 1, :, 1] = parameters['g_cx_th'] * np.dot(self.W_cx_th, self.S_NMDA[i_t - 1, :, 1])
                self.I_th_total[i_t-1,:,:] = self.I_cx_th[i_t-1,:,:] + self.I_0_th + self.I_ext_th[i_t-1,:,:]+ self.I_noise_th  # TODO: implement I noise th


            # Update the firing rates of the two excitatory populations.
            self.R[i_t, :, :2] = self.R[i_t - 1, :, :2] + self.dt * current_to_frequency(self.I_total[i_t - 1, :, :2],
                                                                                         'E', parameters) / parameters[
                                                              'tau_rates'] \
                                 - self.dt * self.R[i_t - 1, :, :2] / parameters['tau_rates']

            # Update the firing rates of the inhibitory population.
            self.R[i_t, :, 2] = self.R[i_t - 1, :, 2] + self.dt * current_to_frequency(self.I_total[i_t - 1, :, 2], 'I',
                                                                                       parameters) / parameters[
                                                            'tau_rates'] \
                                - self.dt * self.R[i_t - 1, :, 2] / parameters['tau_rates']

            # Update the NMDA synapses
            self.S_NMDA[i_t, :, :2] = self.S_NMDA[i_t - 1, :, :2] + self.dt * NMDA_deriv(self.S_NMDA[i_t - 1, :, :2],
                                                                                         self.R[i_t, :, :2], parameters)
            
            # Update the GABA synapses
            self.S_GABA[i_t, :, 2] = self.S_GABA[i_t - 1, :, 2] + self.dt * GABA_deriv(self.S_GABA[i_t - 1, :, 2],
                                                                                       self.R[i_t, :, 2], parameters)
            # make sure SNMDA is smaller than 1. 
            for k in range(self.n_areas):
                if self.S_NMDA[i_t, k, 0] > 1:
                    self.S_NMDA[i_t, k, 0] = 1
                if self.S_NMDA[i_t, k, 1] > 1:
                    self.S_NMDA[i_t, k, 1] = 1
            
            if self.thcxmodel==True:
                # update R and S varible for th areas.
                self.R_th[i_t,:,:] = self.R_th[i_t-1,:,:] + self.dt*current_to_frequency(self.I_th_total[i_t-1,:,:],'th',parameters)/parameters['tau_rates'] \
                                    - self.dt*self.R_th[i_t-1, :,:]/parameters['tau_rates']
                self.S_AMPA_th[i_t,:,:] = self.S_AMPA_th[i_t-1,:,:] + self.dt*AMPA_th_deriv(self.S_AMPA_th[i_t-1,:,:],self.R_th[i_t,:,:],parameters)
                # self.D_th[i_t,:,:] = self.D_th[i_t-1,:,:] + self.dt*(-self.parameters['p_D']*self.D_th[i_t-1,:,:]*self.R_th[i_t,:,:]+
                #                                                      (1-self.D_th[i_t-1,:,:])/self.parameters['tau_D'])

        print('sim done.')
        if vocal_prompt ==True:
            os.system("say 'simulation finished' &")

    def save_delay_activity(self, dumppath, saveoutput):
        # save (early) delay activity
        baselinestarttime = self.stim_on - 0.5 * brian2.second
        baselineendtime = self.stim_on

        earlydelay = 2 * brian2.second
        earlyPAstarttime = self.stim_off + earlydelay - 0.5 * brian2.second
        earlyPAendtime = self.stim_off + earlydelay + 0.5 * brian2.second
        PAstarttime = self.trial_length - 2 * brian2.second
        PAendtime = self.trial_length - 0.5 * brian2.second
        earlypersistentactlist = list()
        self.persistentactlist = list()
        self.baselineactlist = list()
        self.persistentInhactlist = list()

        self.LRtoElist = list()
        self.LRtoIlist = list()
        self.IalltoElist = []
        self.localEtoElist = []
        self.localItoElist = []
        self.S_NMDAlist = []
        self.THtoElist = []
        self.THtoIlist = []
        for i in range(1, self.num_areas + 1):
            baselineVecE1 = self.R[
                np.arange(int(baselinestarttime / self.dt), int(baselineendtime / self.dt), 1), i - 1, 0]
            baselineVecE2 = self.R[
                np.arange(int(baselinestarttime / self.dt), int(baselineendtime / self.dt), 1), i - 1, 1]
            baselineLRtoE = self.I_LR_NMDA[
                np.arange(int(baselinestarttime / self.dt), int(baselineendtime / self.dt), 1), i - 1, 0]
            baselineLRtoI = self.I_LR_NMDA[
                np.arange(int(baselinestarttime / self.dt), int(baselineendtime / self.dt), 1), i - 1, 2]

            earlypersistentVec = self.R[
                np.arange(int(earlyPAstarttime / self.dt), int(earlyPAendtime / self.dt), 1), i - 1, 0]
            persistentVecE1 = self.R[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]
            persistentVecE2 = self.R[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 1]
            persistentVecI = self.R[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 2]

            LRtoEVec = self.I_LR_NMDA[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]
            LRtoIVec = self.I_LR_NMDA[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 2]
            IalltoEVec = self.I_total[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]
            localEtoEVec = self.I_local_NMDA[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]
            localItoEVec = self.I_local_GABA[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]

            S_NMDAVec = self.S_NMDA[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]
            if self.thcxmodel:
                THtoEVec = self.I_th_cx[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]
                THtoEmean = np.mean(THtoEVec)
                self.THtoElist.append(THtoEmean / brian2.nA)

                THtoIVec = self.I_th_cx[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 2]
                THtoImean = np.mean(THtoIVec)
                self.THtoIlist.append(THtoImean / brian2.nA)

            LRweeVec = self.W_E*self.parameters['mu_EE']# calculate the longrange wee TODO
            LRwieVec = self.W_I*self.parameters['mu_IE']

            earlypersistentact = np.mean(earlypersistentVec)
            persistentact = (np.mean(persistentVecE1) + np.mean(persistentVecE2)) / 2 + abs(np.mean(persistentVecE1) - np.mean(persistentVecE2)) / 2
            baselineact = (np.mean(baselineVecE1) + np.mean(baselineVecE2)) / 2 + abs(np.mean(baselineVecE1) - np.mean(baselineVecE2)) / 2
            persistentInhact = np.mean(persistentVecI)

            LRtoE = np.mean(LRtoEVec)
            LRtoI = np.mean(LRtoIVec)
            IalltoE = np.mean(IalltoEVec)
            localEtoE = np.mean(localEtoEVec)
            localItoE = np.mean(localItoEVec)
            S_NMDA_Mean = np.mean(S_NMDAVec)

            earlypersistentactlist.append(earlypersistentact / brian2.Hz)
            self.persistentactlist.append(persistentact / brian2.Hz)
            self.baselineactlist.append(baselineact/brian2.Hz)
            self.persistentInhactlist.append(persistentInhact)
            
            self.LRtoElist.append(LRtoE / brian2.nA)
            self.LRtoIlist.append(LRtoI / brian2.nA)
            self.IalltoElist.append(IalltoE/ brian2.nA)
            self.localEtoElist.append(localEtoE / brian2.nA)
            self.localItoElist.append(localItoE / brian2.nA)

            self.S_NMDAlist.append(S_NMDA_Mean)
            #     print(earlypersistentactlist)
            #     print(persistentactlist)
        # save format:  attractor_counting_VISp_PL_0.1nA
        savefilename = dumppath + 'attractor_counting_' + '_'.join(self.inputareas) + '_' + '{:.2f}'.format(
            self.stim_strength/brian2.pA) + '.txt'

        if saveoutput == True:
            with open(savefilename, 'w') as f:
                json.dump([self.persistentactlist, self.S_NMDAlist], f)
                print('file saved:' + savefilename)

                #     with open(filename) as filename:

                #     if beta_search ==True:
                #         with open(dumppath+'/newbetasearch__'+str(beta_pref)+'.txt','wb') as filename:
                #             pickle.dump([earlypersistentactlist,persistentactlist], filename)
        # if local_optoinh_search ==True:
        #     with open(dumppath+'/newOptoInhSearch__'+inh_area+'.txt','wb') as filename:
        #         pickle.dump([earlypersistentactlist,persistentactlist], filename)

        # else:
        #     with open(dumppath+'/new_defaultmodel'+'.txt','wb') as filename:
        #         pickle.dump([earlypersistentactlist,persistentactlist], filename)
        return self.persistentactlist, self.baselineactlist, self.LRtoElist, self.LRtoIlist


    def save_th_delay_activity(self, dumppath, saveoutput):
        # save (early) delay activity
        baselinestarttime = self.stim_on - 1.5 * brian2.second
        baselineendtime = self.stim_on

        PAstarttime = self.trial_length - 2 * brian2.second
        PAendtime = self.trial_length - 0.5 * brian2.second
        self.th_persistentactlist = list()
        for i in range(1, self.num_th_areas + 1):
            baselineVecE1 = self.R_th[
                np.arange(int(baselinestarttime / self.dt), int(baselineendtime / self.dt), 1), i - 1, 0]
            baselineVecE2 = self.R_th[
                np.arange(int(baselinestarttime / self.dt), int(baselineendtime / self.dt), 1), i - 1, 1]
            persistentVec = self.R_th[np.arange(int(PAstarttime / self.dt), int(PAendtime / self.dt), 1), i - 1, 0]
            persistentact = np.mean(persistentVec)
            baselineact = np.max( (np.mean(baselineVecE1), np.mean(baselineVecE2)) )



            self.th_persistentactlist.append(persistentact / brian2.Hz)

        # save format:  attractor_counting_VISp_PL_0.1nA
        savefilename = dumppath + 'th_attractor_counting_' + '_'.join(self.inputareas) + '_' + str(
            self.stim_strength/brian2.pA) + '.txt'

        if saveoutput == True:
            with open(savefilename, 'w') as filename:
                json.dump(self.th_persistentactlist, filename)
                print('file saved:' + savefilename)

                #     with open(filename) as filename:

                #     if beta_search ==True:
                #         with open(dumppath+'/newbetasearch__'+str(beta_pref)+'.txt','wb') as filename:
                #             pickle.dump([earlypersistentactlist,persistentactlist], filename)
        # if local_optoinh_search ==True:
        #     with open(dumppath+'/newOptoInhSearch__'+inh_area+'.txt','wb') as filename:
        #         pickle.dump([earlypersistentactlist,persistentactlist], filename)

        # else:
        #     with open(dumppath+'/new_defaultmodel'+'.txt','wb') as filename:
        #         pickle.dump([earlypersistentactlist,persistentactlist], filename)
        return self.th_persistentactlist

    def generatePAdf(self, PAthreshold):
        p = self.parameters
        self.persistentact_df = pandas.DataFrame()
        self.persistentact_df['Acronym'] = self.area_list
        self.persistentact_df['Hierarchy'] = np.array(self.hierarchy_df)
        self.persistentact_df['PVdensity'] = np.array(self.normPVgrad_df)
        self.persistentact_df['SSTdensity'] = np.array(self.normSSTgrad_df)

        pv_s = np.array(self.normPVgrad_df)
        pv_hat1 = p['g_E_I'] * (p['J_G_EI_min'] + p['JEI_scaling_factor'] * pv_s)
        pv_hat2 = p['g_Iself'] * (p['J_G_II_min'] + p['JII_scaling_factor'] * pv_s) + \
                  p['g_2'] / (p['c_I'] * p['gamma_GABA'] * p['tau_GABA'])
        # pv_const = p['g_2'] / (p['c_I'] * p['gamma_GABA'] * p['tau_GABA']) * (1 + 0 * pv_s)
        # pv_hat2_noconst = p['g_Iself'] * (p['J_G_II_min'] + p['JII_scaling_factor'] * pv_s)
        pv_hat = pv_hat1 / pv_hat2
        self.persistentact_df['PV_hat'] = pv_hat

        # self.persistentact_df['inDegree'] = inDegree
        # self.persistentact_df['outDegree'] = outDegree
        self.persistentact_df['persistentact'] = self.persistentactlist
        self.persistentact_df['baselineact'] = self.baselineactlist
        self.persistentact_df['persistentactBinary'] = np.array(self.persistentactlist) > PAthreshold
        self.persistentact_df['S_NMDA'] = self.S_NMDAlist
        self.persistentact_df['IalltoE'] = self.IalltoElist
        self.persistentact_df['localEtoE'] = self.localEtoElist
        self.persistentact_df['localItoE'] = self.localItoElist

        if self.thcxmodel:
            self.persistentact_df['THtoE'] = self.THtoElist
            self.persistentact_df['THtoI'] = self.THtoIlist
        self.persistentact_df['LRtoE'] = self.LRtoElist  # nA
        self.persistentact_df['LRtoI'] = self.LRtoIlist  # nA
        # persistentact_df['regionNumber'] = numberList  # used for generate firing rate map using freesurfer.
        # TODO: check the instrength from persistent areas. or from hub areas.

        self.persistentact_df_sort = self.persistentact_df.sort_values(by='PVdensity').reset_index(drop=True)

        return  self.persistentact_df


    def generate_th_PAdf(self, th_PAthreshold, th_hierarchy):
        if self.thcxmodel==True:
            self.th_persistentact_df = pandas.DataFrame()
            self.th_persistentact_df['Acronym'] = self.thal_areas_list
            self.th_persistentact_df['th_Hierarchy']  = th_hierarchy
            self.th_persistentact_df['th_persistentact'] = self.th_persistentactlist
            self.th_persistentact_df['th_persistentactBinary'] = np.array(self.th_persistentactlist) > th_PAthreshold
        else:
            print('thcxmodel not True')
        self.th_persistentact_df_sort = self.th_persistentact_df.sort_values(by='th_Hierarchy').reset_index(drop=True)
        return self.th_persistentact_df

##  need to rewrite
    def plotFRvsHier(self,annotation_adjust, showLabel, dotSize, figureSize, fontSize, saveFig, fileName):
        persistentact_df_sort = self.persistentact_df_sort

        div = self.parameters['div']
        div_name_list = self.parameters['div_name_list']
        div_color_list = self.parameters['div_color_list']

        fig = plt.figure(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
        plt.rcParams.update({'font.size': fontSize})
        # plot FR vs hierarchy
        plt.scatter(persistentact_df_sort['Hierarchy'], persistentact_df_sort['persistentact'],
                    label='hier,FR',linewidths=dotSize)
        # plt.xticks(range(n_areas),persistentact_df_sort['Acronym'], rotation='70')

        ax = plt.gca()

        if showLabel:
            texts = []
            # plt.ylim(-6.5,65) # leave enough space for annotation.
            for i in range(self.n_areas):
                acr = persistentact_df_sort['Acronym'][i]
                if acr in self.area_list:
                    for div_name, div_color in zip(div_name_list, div_color_list):
                        if acr in div[div_name]:
                            texts += [ax.text(persistentact_df_sort['Hierarchy'][i],
                                              persistentact_df_sort['persistentact'][i], acr,
                                              color=div_color, fontsize=fontSize*0.4)]
            # # use adjust library to adjust the position of annotations.
            if annotation_adjust:
                adjust_text(texts, persistentact_df_sort['Hierarchy'], persistentact_df_sort['persistentact'],
                            ax=ax, precision=0.001,
                            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
                            arrowprops=dict(arrowstyle='-', color='gray', alpha=.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('Hierarchy')
        plt.ylabel('Rate (Hz)')
        # plt.ylim([-1,8])

        corr, pvalue = sp.stats.pearsonr(persistentact_df_sort['Hierarchy'],
                                         persistentact_df_sort['persistentact'])
        plt.title('r=' + str(round(corr, 2)) + ', p=' + f"{pvalue:.2E}")
        print(f"{pvalue:.2E}")
        #save figure
        if saveFig:
            fig.savefig('figure/' + fileName, dpi=300, bbox_inches='tight',transparent=True)

        # plt.savefig('figure/neurondensity_grad.png',dpi=80,bbox_inches='tight')

        # plt.scatter(np.array(hierarchy_df), persistentactlist)
        # print(persistentact_df_sort)

    def plotFRvsPV(self, annotation_adjust, showLabel, dotSize, figureSize, fontSize, saveFig, fileName):

        persistentact_df = self.persistentact_df
        persistentact_df_sort = persistentact_df.sort_values(by='PVdensity').reset_index(drop=True)

        div = self.parameters['div']
        div_name_list = self.parameters['div_name_list']
        div_color_list = self.parameters['div_color_list']

        fig = plt.figure(figsize=figureSize, dpi=300, facecolor='w', edgecolor='k')
        plt.rcParams.update({'font.size': fontSize})
        # plot FR vs hierarchy
        plt.scatter(persistentact_df_sort['PVdensity'], persistentact_df_sort['persistentact'],
                    label='hier,PV',linewidths=dotSize)
        # plt.xticks(range(n_areas),persistentact_df_sort['Acronym'], rotation='70')

        ax = plt.gca()
        # plt.ylim(-6.5,65) # leave enough space for annotation.
        if showLabel:
            texts = []
            for i in range(self.n_areas):
                acr = persistentact_df_sort['Acronym'][i]
                if acr in self.area_list:
                    for div_name, div_color in zip(div_name_list, div_color_list):
                        if acr in div[div_name]:
                            texts += [ax.text(persistentact_df_sort['PVdensity'][i],
                                              persistentact_df_sort['persistentact'][i], acr,
                                              color=div_color, fontsize=fontSize*0.4)]

            # # use adjust library to adjust the position of annotations.
            if annotation_adjust:
                adjust_text(texts, persistentact_df_sort['PVdensity'], persistentact_df_sort['persistentact'],
                            ax=ax, precision=0.001,
                            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
                            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
                            arrowprops=dict(arrowstyle='-', color='gray', alpha=.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.setp(ax.spines.values(), linewidth=3)
        # ax.xaxis.set_tick_params(width=7)
        # ax.yaxis.set_tick_params(width=7)

        plt.xlabel('PVdensity')
        plt.ylabel('Rate (Hz)')
        # plt.ylim([-1,8])

        corr, pvalue = sp.stats.pearsonr(persistentact_df_sort['PVdensity'],
                                         persistentact_df_sort['persistentact'])
        plt.title('r=' + str(round(corr, 2)) + ', p=' + f"{pvalue:.2E}")
        print(f"{pvalue:.2E}")

        # save figure
        if saveFig:
            plt.savefig('figure/' + fileName, dpi=300, bbox_inches='tight',transparent=True)
        # plt.savefig('figure/neurondensity_grad.png',dpi=80,bbox_inches='tight')

        # plt.scatter(np.array(hierarchy_df), persistentactlist)
        # print(persistentact_df_sort)


    def plotFRallarea(self,plot_interneuron=False, ylimit=80, figureSize=(30, 250),  savefig=False, figfilename='FRallcortex.pdf'):
        # generate firing rate plots (frplots) for all areas.
        # fig = plt.figure(figsize=(30, 250), dpi=80, facecolor='w', edgecolor='k')
        fig = plt.figure(figsize=figureSize, dpi=80, facecolor='w', edgecolor='k')
        # start plot the firing rate at 0.5s avoiding the early noisy part
        start_time = 0.5  # seconds
        end_time = self.trial_length / brian2.second  # seconds
        # end_time = 12
        plt.rcParams.update({'font.size': 20})

        for i in range(1, self.num_areas + 1):
            # making plots, 5 panel per row
            ax = plt.subplot(self.num_areas, 5, i)
            # specify the area name
            ax.set_title(self.area_list[i - 1])
            plt.subplots_adjust(hspace=1)

            # Plot the rates for the E1&E2
            plt.plot(np.arange(start_time * brian2.second, end_time * brian2.second, self.dt),
                     self.R[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 0],
                     color='green')
            plt.plot(np.arange(start_time * brian2.second, end_time * brian2.second, self.dt),
                     self.R[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 1],
                     color='orange')
            # Toggle: plot the rates for the PV or not plot
            if plot_interneuron ==True:
                plt.plot(np.arange(start_time * brian2.second, end_time * brian2.second, self.dt),
                     self.R[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 2])

            # Plot the stimulation duration bar
            #     if end_time<21:
            #         plt.plot([stim_on,stim_off],[np.max(R[:,i-1,:]+0.1*np.max(R[:,i-1,:])),np.max(R[:,i-1,:]+0.1*np.max(R[:,i-1,:]))],color='black',linewidth=5.0)

            # hide the frames
            axes = plt.gca()
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)

            plt.legend(['E1', 'E2'], fontsize=16, frameon=False)
            plt.xlabel('Time (s)', fontname='Arial', fontsize=16)
            plt.ylabel('Rate (Hz)', fontname='Arial', fontsize=16)
            # if noise_network == True:
            #     plt.ylim(0, 1)
            # else:
            plt.ylim(0, ylimit)
            plt.yticks([0, ylimit/4, ylimit/2, ylimit/4*3])

        # save figure
        folder = self.parameters['figurefolder']
        if savefig ==True:
            plt.savefig(folder+figfilename,dpi=80,bbox_inches='tight',transparent=True)

    def diffAnalyze(self):
        self.diff_I_local_NMDA = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        self.diff_I_local_NMDA[0, :, :] = 0

        self.diff2_I_local_NMDA = np.zeros((self.num_iterations, self.num_areas, self.num_pops)) * brian2.pA
        self.diff2_I_local_NMDA[0, :, :] = 0
        self.diff2_I_local_NMDA[1, :, :] = 0
        self.diff2_I_local_NMDA[2, :, :] = 0

        for i_t in range(4, self.num_iterations):
            self.diff_I_local_NMDA[i_t - 1, :, :] = self.I_local_NMDA[i_t - 1, :, :] - self.I_local_NMDA[i_t - 2, :, :]
            self.diff2_I_local_NMDA[i_t - 1, :, :] = self.I_local_NMDA[i_t - 1, :, :] - 2 * self.I_local_NMDA[i_t - 2, :, :] +\
                                    self.I_local_NMDA[i_t - 3, :, :]


    def plotTimeSeries(self, targetVariable, plot_interneuron=False, ylimit=80, savefig=False, figfilename='FRallcortex.pdf'):
            # generate time series plots (frplots) for all areas.
            fig = plt.figure(figsize=(30, 250), dpi=80, facecolor='w', edgecolor='k')
            # start plot the firing rate at 0.5s avoiding the early noisy part
            start_time = 0.5  # seconds
            end_time = self.trial_length / brian2.second  # seconds
            # end_time = 12
            plt.rcParams.update({'font.size': 20})

            Xvalues = np.arange(start_time * brian2.second, end_time * brian2.second, self.dt)
            if targetVariable == 'firing_rate':
                Yvalue = self.R
            elif targetVariable == 'current_local_ex':
                Yvalue = self.I_local_NMDA
            elif targetVariable == 'current_local_inh':
                Yvalue = self.I_local_GABA
            elif targetVariable == 'current_LR':
                Yvalue = self.I_LR_NMDA
            elif targetVariable == 'current_cross':
                Yvalue = self.I_local_cross_NMDA
            elif targetVariable == 'diff_current_local_ex':
                Yvalue = self.diff_I_local_NMDA
            elif targetVariable == 'diff2_current_local_ex':
                Yvalue = self.diff2_I_local_NMDA
            else:
                raise ValueError

            for i in range(1, self.num_areas + 1):
                # making plots, 5 panel per row
                ax = plt.subplot(self.num_areas, 5, i)
                # specify the area name
                ax.set_title(self.area_list[i - 1])
                plt.subplots_adjust(hspace=1)

                # Plot the rates for the E1&E2
                plt.plot(Xvalues,
                         Yvalue[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 0],
                         color='green')
                plt.plot(Xvalues,
                         Yvalue[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 1],
                         color='orange')
                # Toggle: plot the rates for the PV or not plot
                if plot_interneuron == True:
                    plt.plot(Xvalues,
                             Yvalue[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 2])

                # Plot the stimulation duration bar
                #     if end_time<21:
                #         plt.plot([stim_on,stim_off],[np.max(R[:,i-1,:]+0.1*np.max(R[:,i-1,:])),np.max(R[:,i-1,:]+0.1*np.max(R[:,i-1,:]))],color='black',linewidth=5.0)

                # hide the frames
                axes = plt.gca()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)

                plt.legend(['E1', 'E2'], fontsize=16, frameon=False)
                plt.xlabel('Time (s)', fontname='Arial', fontsize=16)
                if targetVariable == 'firing_rate':
                    plt.ylabel('Rate (Hz)', fontname='Arial', fontsize=16)
                else:
                    plt.ylabel('current (A)', fontname='Arial', fontsize=16)
                # if noise_network == True:
                #     plt.ylim(0, 1)
                # else:
                plt.ylim(0, ylimit)
                plt.yticks([0, ylimit / 4, ylimit / 2, ylimit / 4 * 3])

            # save figure
            folder = self.parameters['figurefolder']
            if savefig == True:
                plt.savefig(folder + figfilename, dpi=80, bbox_inches='tight', transparent=True)



                # plt.savefig('figure/LSM_T1T2_PVgrad_SSp_10s_stronger_gIE_final.png',dpi=80,bbox_inches='tight')
                # plt.savefig('figure/LSM_T1T2_PVgrad_SSp_10s_stronger_gIE_ILA_inhibited.png',dpi=80,bbox_inches='tight')
                # plt.savefig('figure/LSM_noisesimulation_TC.png',xdpi=80,bbox_inches='tight')
                # plt.savefig('figure/LSM_T1T2_PVgrad_SSp_10s_stronger_gIE_disconnected.png',dpi=80,bbox_inches='tight')

    def plotFRthalarea(self,ylimit =80, savefig = False, figfilename='FRallthalamus.pdf'):
        # generate firing rate plots (frplots) for all areas.
        fig = plt.figure(figsize=(30, 250), dpi=80, facecolor='w', edgecolor='k')
        # start plot the firing rate at 0.5s avoiding the early noisy part
        start_time = 0.5  # seconds
        end_time = self.trial_length / brian2.second  # seconds
        # end_time = 12
        plt.rcParams.update({'font.size': 20})

        for i in range(1, self.num_th_areas + 1):
            # making plots, 5 panel per row
            ax = plt.subplot(self.num_th_areas, 5, i)
            # specify the area name
            ax.set_title(self.thal_areas_list[i - 1])
            plt.subplots_adjust(hspace=1)

            # Plot the rates for the E1&E2
            plt.plot(np.arange(start_time * brian2.second, end_time * brian2.second, self.dt),
                     self.R_th[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 0],
                     color='green')
            plt.plot(np.arange(start_time * brian2.second, end_time * brian2.second, self.dt),
                     self.R_th[np.arange(int(start_time / self.dt), int(end_time / self.dt), 1), i - 1, 1],
                     color='orange')

            # Plot the stimulation duration bar
            #     if end_time<21:
            #         plt.plot([stim_on,stim_off],[np.max(R[:,i-1,:]+0.1*np.max(R[:,i-1,:])),np.max(R[:,i-1,:]+0.1*np.max(R[:,i-1,:]))],color='black',linewidth=5.0)

            # hide the frames
            axes = plt.gca()
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)

            plt.legend(['E1', 'E2'], fontsize=16, frameon=False)
            plt.xlabel('Time (s)', fontname='Arial', fontsize=16)
            plt.ylabel('Rate (Hz)', fontname='Arial', fontsize=16)
            # if noise_network == True:
            #     plt.ylim(0, 1)
            # else:
            plt.ylim(0, ylimit)
            plt.yticks([0, ylimit/4, ylimit/2, ylimit/4*3])

        # save figure
        folder = self.parameters['figurefolder']
        if savefig == True:
            plt.savefig(folder + figfilename, dpi=80, bbox_inches='tight',transparent = True)

                # plt.savefig('figure/LSM_T1T2_PVgrad_SSp_10s_stronger_gIE_final.png',dpi=80,bbox_inches='tight')
                # plt.savefig('figure/LSM_T1T2_PVgrad_SSp_10s_stronger_gIE_ILA_inhibited.png',dpi=80,bbox_inches='tight')
                # plt.savefig('figure/LSM_noisesimulation_TC.png',xdpi=80,bbox_inches='tight')
                # plt.savefig('figure/LSM_T1T2_PVgrad_SSp_10s_stronger_gIE_disconnected.png',dpi=80,bbox_inches='tight')

