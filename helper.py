
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pickle
from matplotlib.colors import LogNorm
from copy import deepcopy

#
# from allensdk.api.queries.ontologies_api import OntologiesApi
# from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
# from mcmodels.core import VoxelModelCache

# Load original Harris hierarchy
def load_hierarchy(filepath, hierfilename, areas):

    # filepath is where the file stored
    # hierfilename is the filename of hierarchy form
    # areas is a llist of interested areas
    # the function returns a data frame of hieararchy index for given areas.
    with open(filepath + hierfilename, 'rb') as f:
        harrishier_df = pandas.read_csv(f ,sep=',' ,names=['area' ,'harrishier'])
    harrishier_df_sorted = harrishier_df.sort_values(by='area')
    harrishier_df_sorted.set_index('area' ,inplace=True)
    harrishier_sorted=[ harrishier_df_sorted.loc[area, 'harrishier'] for area in areas]
    harrishier_sorted = np.array(harrishier_sorted).astype(float)
    harrishierarchy = (harrishier_sorted- harrishier_sorted.min())/(
    harrishier_sorted.max()- harrishier_sorted.min())
    harrishierarchy = np.array(harrishierarchy)
    harrishierarchy = harrishierarchy.reshape((-1,1)) # AUDpo SSp-un FRP are not inclued in AIBS connectivity, but included in T1T2 data.
    harrishierarchy_df = pandas.DataFrame(harrishierarchy,index=areas, columns=[ 'hierarchy index'])
    harrishierarchy_df.sort_values(by='hierarchy index')
    #     harrishierarchy_df.get_value('PL',col='hierarchy index')
    return harrishierarchy, harrishierarchy_df

# # load the latest harris connectivity 43*43.
# def load_connectivity():
#     mcc = MouseConnectivityCache(manifest_file='connectivity/mouse_connectivity_manifest.json')
#     # The manifest file is a simple JSON file that keeps track of all of
#     # the data that has already been downloaded onto the hard drives.
#     # If you supply a relative path, it is assumed to be relative to your
#     # current working directory.
#
#     # grab the StructureTree instance
#     structure_tree = mcc.get_structure_tree()
#
#     oapi = OntologiesApi()
#
#     # get the ids of all the structure sets in the tree
#     structure_set_ids = structure_tree.get_structure_sets()
#
#     # download and cache the latest voxel model
#     # this method returns a tuple with object types:
#     # (VoxelConnectivityArray, Mask, Mask)
#     cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
#
#     # extracting the cortical subnetwork. get the normalized connectvity density for ipsi side.
#     normalized_connection_density = cache.get_normalized_connection_density()
#     cortex_matrix = normalized_connection_density['ipsi']  # .loc[cortex_ids_int][cortex_ids_str]
#     mat = np.array(cortex_matrix)
#
#     mat = mat.transpose() #  ake a transpose of the mat since what the paper used in the transponsed version.
#
#     # cortex
#     structures_cortex = structure_tree.get_structures_by_set_id([688152357])
#     cortex_ids_int = [s['id'] for s in structures_cortex]  # store as a int
#     cortex_ids_str = [str(s['id']) for s in structures_cortex]  # store as a str
#     cortex_acr = [s['acronym'] for s in structures_cortex]
#     cortex_names = [s['name'] for s in structures_cortex]
#
#     cortical_area_number = np.size(cortex_acr)
#     print(cortical_area_number)
#
#     W = mat[0:cortical_area_number, 0:cortical_area_number]  # extract the cortical areas
#
#     return W, cortex_acr


def plot_connectivity(W_img, areas, minimum_value_to_show = 1e-6,savefig=False):
    plt.figure()
    plt.imshow(W_img, norm=LogNorm(vmin=minimum_value_to_show, vmax=W_img.max()))
    plt.colorbar()
    plt.ylabel('Target')
    plt.xlabel('Source')
    n_areas = np.size(areas)
    plt.xticks(list(range(n_areas)),areas,rotation=90,fontname='Georgia',fontsize=7)
    plt.yticks(list(range(n_areas)),areas,fontname='Georgia',fontsize=7)
    plt.grid(False)
    if savefig==True:
        plt.savefig('figure/mouse_original_connection2_log.pdf',dpi=200,bbox_inches='tight')

def generate_connectivity(W, cortex_acr, W_cxth, W_thcx, thal_acr, imgout=True):

    full_areas = cortex_acr
    n_fullarea = len(full_areas)
    np.fill_diagonal(W, 0)

    # Remove areas like AUDv, ECT, GU and PERI from connectivity.
    # they don't have data that passed thresholding.(connection strength>10^-1.5)
    area_unsort = deepcopy(full_areas)

    # remove areas not having hierarchy data.
    # area_unsort.remove('AUDv')
    # area_unsort.remove('ECT')
    # area_unsort.remove('GU')
    # area_unsort.remove('PERI')
    # # remove SSp-un and VISC for the same reason, according to the published paper.
    # area_unsort.remove('SSp-un')
    # area_unsort.remove('VISC')

    # areaidxlist = np.delete(np.arange(len(full_areas)), [cortex_acr.index('AUDv'), cortex_acr.index('ECT'),
    #                                                      cortex_acr.index('GU'), cortex_acr.index('PERI'),
    #                                                      cortex_acr.index('SSp-un'), cortex_acr.index('VISC')])
    # print(areaidxlist)
    #
    areaidxlist = np.arange(len(full_areas))

    # extract the connectivity for list of areas.
    W = W[areaidxlist, :][:, areaidxlist]
    print(W.shape)

    # sort the areas according to their spelling.
    cortex_sorting_index = np.argsort(area_unsort)
    area_sort = [area_unsort[i] for i in cortex_sorting_index]
    n_areas = np.size(area_sort)
    print(area_sort)
    print(np.argsort(area_sort))
    print(cortex_sorting_index)
    conn_cxcx = W[cortex_sorting_index, :][:, cortex_sorting_index]
    # conn_cxcx = W[np.argsort(area_sort), :][:, np.argsort(area_sort)]

    W_cxth = W_cxth[:,:][:,areaidxlist]
    W_thcx = W_thcx[areaidxlist,:][:,:]
    print(W_cxth.shape)

    conn_cxth = W_cxth[:,:][:,cortex_sorting_index]
    conn_thcx = W_thcx[cortex_sorting_index,:][:,:]

    area_list_sort = area_sort # sort accroding to area name.
    thal_list_sort = thal_acr # before known results, I will just sort the thal according to the order in data.
    if imgout == True:
        plt.figure()
        plt.imshow(conn_cxcx)
        plt.xticks(list(range(n_areas)), area_sort, rotation=90, fontname='Georgia', fontsize=7)
        plt.yticks(list(range(n_areas)), area_sort, fontname='Georgia', fontsize=7)
        plt.colorbar()

        plt.figure()
        plt.imshow(conn_cxth)
        plt.xticks(list(range(n_areas)), area_sort, rotation=90, fontname='Georgia', fontsize=7)
        plt.xlabel('source')
        plt.yticks(list(range(len(thal_acr))), thal_acr, fontname='Georgia', fontsize=7)
        plt.ylabel('target')
        plt.colorbar()

        plt.figure()
        plt.imshow(conn_thcx)
        plt.xticks(list(range(len(thal_acr))), thal_acr, rotation=90, fontname='Georgia', fontsize=7)
        plt.xlabel('source')
        plt.yticks(list(range(n_areas)), area_sort, fontname='Georgia', fontsize=7)
        plt.ylabel('target')
        plt.colorbar()

        plt.show()

    return conn_cxcx, conn_cxth, conn_thcx, area_list_sort, thal_list_sort


def generate_random_connectivity(N, imgout=True):
    conn_cxcx = np.random.rand(N, N)# the matrix will be normalized afterwards.
    if imgout == True:
        plt.imshow(conn_cxcx)
#         plt.xticks(list(range(N)),areas,rotation=90,fontname='Georgia',fontsize=7)
#         plt.yticks(list(range(N)),areas,fontname='Georgia',fontsize=7)
        plt.colorbar()
        plt.show()
    return conn_cxcx


def load_celldensity(filepath, celldensityfilename, old_areas):
    # PV density is normalized by total neuron density.
    with open(filepath + '/' + celldensityfilename, 'rb') as f:
        cell_df = pandas.read_csv(f, sep=',')

    cell_df_sorted = cell_df.sort_values(by='Neurons', axis=0)
    cell_df_sorted.set_index('Acronym', inplace=True)

    neurons_sorted = [cell_df_sorted.loc[area, 'Neurons'] for area in old_areas]
    neurons_sorted = np.array(neurons_sorted)
    return neurons_sorted


def load_interneurondensity(filepath, interneuronfilename, old_areas):
    names = ['PV', 'SST', 'VIP', 'Gad2']  # extract certain type of interneuron data
    with open(filepath + '/' + interneuronfilename, 'rb') as f:
        p_kim = pickle.load(f, encoding='latin1')  # use encoding = latin1 to load pickle files from earlier version.
    inh_density_full = np.array([p_kim['pv_list'],
                                 p_kim['sst_list'],
                                 p_kim['vip_list'],
                                 p_kim['gad_list']]).T

    idx = [p_kim['areas'].index(area) for area in old_areas]
    inh_density = inh_density_full[idx, :]

    layers = ['1', '2/3', '5', '6a']  # extract certain layer of data
    inh_den_layers = dict()
    for layer in layers:
        idx = [p_kim['areas'].index(area + layer) for area in old_areas]
        inh_den_layers[layer] = inh_density_full[idx, :]
    inh_den_layers['all'] = inh_density  # extract  density for all layers
    inh_den_layers['5/6'] = (inh_den_layers['5'] + inh_den_layers['6a']) / 2  # average density for layer 5 and layer 6

    PV_23 = inh_den_layers['2/3'][:, 0:1] # extract L2/3 PV density
    SST_23 = inh_den_layers['2/3'][:, 1:2]  # extract L2/3 PV density
    PV_all = inh_den_layers['all'][:, 0:1]  # extract all layer PV # density
    SST_all = inh_den_layers['all'][:, 1:2]
    interneuron_all = np.sum(inh_den_layers['all'], axis=1)   #  TODO: this is a bug, this is not all neuron!
    interneuron_all = interneuron_all.reshape((-1, 1))
    #     neurons_sorted = neurons_sorted.reshape((-1,1))
    #   not use total neuron number due to different data sources.

    # PVnormNeuron = PV_all / interneuron_all
    # SSTnormNeuron = SST_all / interneuron_all
    # normPVgrad = (PVnormNeuron - PVnormNeuron.min()) / (PVnormNeuron.max() - PVnormNeuron.min())
    # normSSTgrad = (SSTnormNeuron - SSTnormNeuron.min()) / (SSTnormNeuron.max() - SSTnormNeuron.min())
    PVgrad = np.array(PV_all).reshape((-1, 1))
    SSTgrad = np.array(SST_all).reshape((-1, 1))

    normPVgrad = (PV_all - PV_all.min()) / (PV_all.max() - PV_all.min())
    normSSTgrad = (SST_all - SST_all.min()) / (SST_all.max() - SST_all.min())

    # normPVgrad = PV_all / PV_all.max()
    # normSSTgrad = SST_all / SST_all.max()

    normPVgrad = np.array(normPVgrad)
    normSSTgrad = np.array(normSSTgrad)
    normPVgrad = normPVgrad.reshape((-1, 1))
    normSSTgrad = normSSTgrad.reshape((-1, 1))

    #     A_ = np.array([PV_all,PVnormNeuron])
    #     print(np.shape(A_))

    PVgrad_df = pandas.DataFrame(PVgrad, index=old_areas, columns=['raw PV density'])
    SSTgrad_df = pandas.DataFrame(SSTgrad, index=old_areas, columns=['raw SST density'])

    normPVgrad_df = pandas.DataFrame(normPVgrad, index=old_areas, columns=['norm PV gradient'])
    normSSTgrad_df = pandas.DataFrame(normSSTgrad, index=old_areas, columns=['norm SST gradient'])

    # areas_with_PA={'AIp','ECT','ORBm','PERI','PL','TEa'}
    # areas_with_activity={'AId','AIp','AIv','AUDd','AUDp','AUDv','GU','ILA','SSp-bfd','VISC','VISal','VISam','VISl','VISp','VISpl','VISpm'}
    # areas_higher_than_SSp={'MOp','SSp-tr','SSp-ll','MOs','PTLp','RSPagl','SSs','VISal','AUDd','AUDp','VISl','VISam','AId','VISC','VISp','ILA','AUDv','AIv','VISpl','VISpm','GU','ORBm','PL','PERI','ECT','TEa','AIp'}
    PVgrad_df.sort_values(by='raw PV density')
    SSTgrad_df.sort_values(by='raw SST density')
    normPVgrad_df.sort_values(by='norm PV gradient')
    normSSTgrad_df.sort_values(by='norm SST gradient')
    return PVgrad_df, SSTgrad_df, normPVgrad_df, normSSTgrad_df

def interneuronDensityProcessing(inputDf):
    # normPVgrad_df_ = deepcopy(normPVgrad_df)
    # normSSTgrad_df_ = deepcopy(normSSTgrad_df)
    #
    # pv_VISa = normPVgrad_df_.at['PTLp', 'norm PV gradient']
    # pv_FRP = normPVgrad_df_.at['ORBm', 'norm PV gradient']  # not sure
    # pv_VISli = normPVgrad_df_.at['VISl', 'norm PV gradient']  # it's also possible VISli belongs to TEa
    # pv_VISrl = normPVgrad_df_.at['PTLp', 'norm PV gradient']
    # pv_VISpor = normPVgrad_df_.at['VISpl', 'norm PV gradient']
    # normPVgrad_df_.loc['VISa'] = pv_VISa
    # normPVgrad_df_.loc['FRP'] = pv_FRP
    # normPVgrad_df_.loc['VISli'] = pv_VISli
    # normPVgrad_df_.loc['VISrl'] = pv_VISrl
    # normPVgrad_df_.loc['VISpor'] = pv_VISpor
    # normPVgrad_df_.drop('PTLp', inplace=True)
    # normPVgrad_df_.sort_index(inplace=True)
    #
    # # areas like AUDv, ECT, GU and PERI, VISC don't have data that passed thresholding.(connection strength>10^-1.5)
    # normPVgrad_df_.drop('AUDv', inplace=True)
    # normPVgrad_df_.drop('ECT', inplace=True)
    # normPVgrad_df_.drop('GU', inplace=True)
    # normPVgrad_df_.drop('PERI', inplace=True)
    # normPVgrad_df_.drop('VISC', inplace=True)
    #
    # SST_VISa = normSSTgrad_df_.at['PTLp', 'norm SST gradient']
    # SST_FRP = normSSTgrad_df_.at['ORBm', 'norm SST gradient']  # not sure
    # SST_VISli = normSSTgrad_df_.at['VISl', 'norm SST gradient']  # it's also possible VISli belongs to TEa
    # SST_VISrl = normSSTgrad_df_.at['PTLp', 'norm SST gradient']
    # SST_VISpor = normSSTgrad_df_.at['TEa', 'norm SST gradient']
    # normSSTgrad_df_.loc['VISa'] = SST_VISa
    # normSSTgrad_df_.loc['FRP'] = SST_FRP
    # normSSTgrad_df_.loc['VISli'] = SST_VISli
    # normSSTgrad_df_.loc['VISrl'] = SST_VISrl
    # normSSTgrad_df_.loc['VISpor'] = SST_VISpor
    # normSSTgrad_df_.drop('PTLp', inplace=True)
    # normSSTgrad_df_.sort_index(inplace=True)
    #
    # normSSTgrad_df_.drop('AUDv', inplace=True)
    # normSSTgrad_df_.drop('ECT', inplace=True)
    # normSSTgrad_df_.drop('GU', inplace=True)
    # normSSTgrad_df_.drop('PERI', inplace=True)
    # normSSTgrad_df_.drop('VISC', inplace=True)
    # return normPVgrad_df_, normSSTgrad_df_

    areaV3toV2 = {'VISa':'PTLp', 'FRP':'PL', 'VISli':'VISl',
        'VISrl':'PTLp', 'VISpor':'VISpl', 'SSp-un': 'SSp-ul'}
    # areaDrop = ['PTLp', 'AUDv', 'ECT', 'GU', 'PERI', 'VISC']
    areaDrop = ['PTLp']
    # inputDf = [PVgrad_df, SSTgrad_df, normPVgrad_df, normSSTgrad_df]
    outputDf = []
    for df in inputDf:
        df_ = deepcopy(df)
        for newArea in areaV3toV2:
            oldArea = areaV3toV2[newArea]
            pvValue = df.at[oldArea, df.columns[0]]
            df_.loc[newArea] = pvValue
        for darea in areaDrop:
            df_.drop(darea, inplace = True)
        df_.sort_index(inplace = True)
        outputDf.append(df_)

    return outputDf

# generate pref targetting matrix
def generate_pref(harrishierarchy, p):
    alpha_pref = p['alpha_pref']
    beta_pref = p['beta_pref']
    h = harrishierarchy
    n = np.size(h)
    pref_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pref_matrix[i, j] = 1 / (1 + np.exp(-beta_pref * (h[i] - h[j]))) \
                                + alpha_pref
            # use logistic function to calculate pref matrix
            # pref matrix term>0.5 FF projections; pref matrix term<0.5 FB projections
    # pref_matrix = pref_matrix.transpose() # this seems to be necessary
    pref_df = pandas.DataFrame(pref_matrix)
    return pref_matrix

def plot_pref(pref_matrix,area_list):
    n = pref_matrix.shape[0]
    pref_plot = pref_matrix
    plt.imshow(pref_plot)
    plt.colorbar()
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.xticks(list(range(n)), area_list, rotation=90, fontsize=7, fontname='Georgia')
    plt.yticks(list(range(n)), area_list, fontsize=7, fontname='Georgia')
    plt.grid(False)
    # plt.savefig('figure/mouseT1T2_hierarchy_FFness.png',dpi=80,bbox_inches='tight')
