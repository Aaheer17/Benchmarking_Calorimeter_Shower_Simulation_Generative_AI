# model_to_color_dict = {'Geant4':'grey',
#                       'CaloDiffusion':'salmon',
#                       'CaloINN':'green',
#                       'CaloScore':'blue',
#                       'CaloDream':'magenta'}

# model_to_color_dict = {'Geant4':'grey',
#                        'CaloDiffFull':'salmon',
#                        'CaloDiffMix':'magenta'}

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import HighLevelFeatures as HLF
import re
from matplotlib import gridspec
from scipy.stats import wasserstein_distance
import configargparse
import jetnet
import random


# Predefined color palette
color_palette = ['orange', 'purple', 'cyan', 'yellow', 'brown']

def random_color(existing_colors):
    # Generate a random color
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    # Keep generating a new color until it's not in the existing colors
    while color in existing_colors:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    return color

def create_model_to_color_dict(models):
    """ Function to create dynamic model to color dictionary """

    model_to_color_dict = {'Geant4':'grey',
                      'CaloDiffusion':'salmon',
                      'CaloINN':'green',
                      'CaloScore':'blue',
                      'CaloDream':'magenta'}
    
    for idx, m in enumerate(models):
        if m not in model_to_color_dict.keys():
            existing_colors = set(model_to_color_dict.values())
            model_to_color_dict[m] = random_color(existing_colors)

    return model_to_color_dict

def write_dict_to_txt(dictionary, filename):
    """ helper function to write a dictionary to a .txt file """

    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def parse_arguments():
    parser = configargparse.ArgumentParser(default_config_files=[])
    parser.add_argument('--dataset_path', type=str, required=False, 
                        default='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/',
                        help='path to generated h5/hdf5 files are stored')
    parser.add_argument('--output_dir',type=str,
                        default='results/')
    parser.add_argument('--binning_file', type=str, required=False, 
                        default='binning_dataset_2.xml',
                        help='path to binning file')
    parser.add_argument('--particle_type', type=str, required=True,default='electron',
                        help='type of the particle being evaluated e.g., photon, pion, electron')
    parser.add_argument('--dataset_num', type=int, required=True, default=2,
                        help='dataset number e.g., 1, 2, 3')
    parser.add_argument('--title_size', type=int, required=False, default=48,
                        help='size of plot title')
    parser.add_argument('--xlabel_size', type=int, required=False, default=44,
                        help='size of xlabels')
    parser.add_argument('--ylabel_size', type=int, required=False, default=44,
                        help='size of ylabels')
    parser.add_argument('--xtick_size', type=int, required=False, default=30,
                        help='size of xtick')
    parser.add_argument('--ytick_size', type=int, required=False, default=30,
                        help='size of ytikc')
    parser.add_argument('--legend_size',type=int,required=False,default=30,
                       help='legend size')
    parser.add_argument('--row',type=int,default=3,
                       help='row size for subplot, for dataset 2 and 3 it should be 3, for pion it is 2 and photon it is 1')
    parser.add_argument('--col',type=int,default=3,
                       help='column size for subplot, for dataset 2 and 3 it should be 3, for pion and photon it is 2')
    parser.add_argument('--metrics',type=str,required=False,default='all',
                        help='type of metrics to evaluate'+\
                        'all--is used for all high level features\' histograms such as layer dist, shower width, center of energy, sparsity, voxel dist, E_ratio'+\
                        'sep-- is used to generate separation power graph'+\
                        'emd-- is used to generate emd graph'+\
                        'fpd-kpd-- is used to generate fpd,kpd values'+\
                        'CFD --is used to generate pearson frobenius norm barplot and histogram'
                        'layer-- is used to generate layer wise energy distribution.'
                       )
    parser.add_argument('--sep_file_path',type=str, help='path to file of separation and emd score')

    args = parser.parse_args()
    return args

def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
    return e, shower

def extract_name_part(file_name):
    """ Use regular expression to extract the desired part of the filename """

    match = re.search(r'_([^_]+)\.h5(?:df5)?$', file_name)
    if match:
        return match.group(1)
    else:
        match = re.search(r'_([^_]+)\.hdf5$', file_name)
        return match.group(1)
    
def iterate_files(directory):
    Es = []
    Showers = []
    model_names = []
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".h5") or filename.endswith(".hdf5"):
            file_path = os.path.join(directory, filename)
            e, shower = file_read(file_path)
            Es.append(e)
            Showers.append(shower)
            name_part = extract_name_part(filename)
            files.append(file_path)
            
            if name_part:
                model_names.append(name_part)
    return model_names, Es, Showers, files

def save_reference(ref_hlf, fname):
    """ Saves high-level features class to file """

    print("Saving file with high-level features.")
    with open(fname, 'wb') as file:
        pickle.dump(ref_hlf, file)
    print("Saving file with high-level features DONE.")

def check_file(given_file, dataset, which=None):
    """ checks if the provided file has the expected structure based on the dataset """

    print("Checking if {} file has the correct form ...".format(
        which if which is not None else 'provided'))
    num_features = {'1-photons': 368, '1-pions': 533, '2': 6480, '3': 40500}[dataset]
    num_events = given_file['incident_energies'].shape[0]
    assert given_file['showers'].shape[0] == num_events, \
        ("Number of energies provided does not match number of showers, {} != {}".format(
            num_events, given_file['showers'].shape[0]))
    assert given_file['showers'].shape[1] == num_features, \
        ("Showers have wrong shape, expected {}, got {}".format(
            num_features, given_file['showers'].shape[1]))

    print("Found {} events in the file.".format(num_events))
    print("Checking if {} file has the correct form: DONE \n".format(
        which if which is not None else 'provided'))

def extract_shower_and_energy(given_file, which):
    """ reads .hdf5 file and returns samples and their energy """

    print("Extracting showers from {} file ...".format(which))
    shower = given_file['showers'][:]
    energy = given_file['incident_energies'][:]
    print("Extracting showers from {} file: DONE.\n".format(which))
    return shower, energy


def initialize_HLFs(path,particle,binning_file):
    Es = []
    Showers = []
    HLFs = []
    model_names, Es, Showers, files = iterate_files(path)

    for i in range(len(model_names)):
        hlf=HLF.HighLevelFeatures(particle, binning_file)
        hlf.Einc = Es[i]
        hlf.CalculateFeatures(Showers[i])
        HLFs.append(hlf)
    return Es, Showers, HLFs, model_names, files


def prepare_high_data_for_classifier(hdf5_file, hlf_class, label):
    """ takes hdf5_file, extracts high-level features, appends label, returns array """

    voxel, E_inc = extract_shower_and_energy(hdf5_file, label)
    nan_indices = np.where(np.isnan(voxel))
          
    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
  
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)], axis=1)
    return ret

def grouping_data(data):
    """
    First summing along the angular bins making it an array of shape(-1,45,radial_bin)
    grouping consecutive 5 layers  
    """

    data=np.sum(data, axis=2)
    data = data.reshape(-1, 9, 5, 9)
    data = data.mean(axis=2)
    #print("in grouping data : ",data.shape)
    return data
