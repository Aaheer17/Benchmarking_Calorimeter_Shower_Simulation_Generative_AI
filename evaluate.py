"""
This code is  partially adopted from CaloChallenge github page. Here is the link for it
https://github.com/CaloChallenge/homepage/blob/main/code/

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import HighLevelFeatures as HLF
from utils import *
import re
from matplotlib import gridspec
from scipy.stats import wasserstein_distance
from evaluate_metrics_helper import *
import configargparse
import jetnet
from compute_metrics_helper import *
#from pearson_frob import *
#from evaluate_range import *

    
def evaluate_metrics_ds_2_3(Es, Showers, HLFs, model_names, files, args, model_to_color_dict):
    """Plot histograms for dataset 2 and 3 """

    #compute layer-wise energy distribution:
    plot_layers(args, model_to_color_dict, Showers, HLFs, Es, model_names)
    
    #Plot mean energy distribution in angular, radial and z direction
    plot_mean_energy(Showers, model_names, model_to_color_dict, args.dataset_num, args.output_dir, args.particle_type)
    
    #compute and plot correlation frobanius norm
    calc_cfd(Showers, model_names, args.output_dir, args.dataset_num, model_to_color_dict)
    
    #plot shower width in eta and phi direction, we grouped five consecutive layers
    plot_SW_group(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict,mode='eta', 
                      ratio=False, row=3, col=3, height=6, width=8, YMAX=100, LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    plot_SW_group(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict,mode='phi', 
                      ratio=False, row=3, col=3, height=6, width=8, YMAX=100, LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    
    #Plot center of energy in eta and phi direction by grouping 5 consecutive layers
    plot_EC_group(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict,'eta',
                      ratio=False, row=3, col=3, height=6, width=8, YMAX=100, LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
        
    plot_EC_group(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict,'phi',
                      ratio=False, row=3, col=3, height=6, width=8, YMAX=100, LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    
    #plot voxel energy dist
    min_energy = 0.5e-3/0.033
     
    plot_cell_dist(Showers, min_energy, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict, 
                      ratio=False, height=6, width=8, YMAX=100, LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    
    #Plot E_ratio
    plot_Etot_Einc_new(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names,model_to_color_dict, 
                      row=1, col=1, height=6, width=8, YMAX=20, LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    #plot_sparsity
    plot_sparsity_group(HLFs, args.dataset_num, args.output_dir,args.particle_type, model_names,model_to_color_dict, width=4,height=3,
                        TITLE_SIZE=args.title_size, XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                                LEGEND_SIZE=args.legend_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)


def evaluate_metrics_ds_1(Es, Showers, HLFs, model_names, files, args, model_to_color_dict):
    """ Plot histograms for dataset 1""" 

    plot_Etot_Einc_new(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names,model_to_color_dict, 
                              row=1, col=1, height=6, width=8, YMAX=25, LEGEND_SIZE=args.legend_size,
                              XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                              TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    min_energy = 10

    plot_cell_dist(Showers, min_energy, args.dataset_num, args.output_dir, args.particle_type, model_names,model_to_color_dict, width=7, height=4,
                       TITLE_SIZE=args.title_size, XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                       LEGEND_SIZE=args.legend_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size, YMAX=3, ratio=False)
    plot_SW(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict, 'eta',
                    row=args.row, col=args.col, height=4, width=6,
                    YMAX=100, LEGEND_SIZE=args.legend_size, XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    plot_SW(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict,'phi',
                    row=args.row, col=args.col, height=4, width=6,
                    YMAX=100, LEGEND_SIZE=args.legend_size, XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size, TICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    
    plot_EC(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict,'eta',
                ratio=False, row=args.row, col=args.col, height=4, width=6,
                YMAX=100, LEGEND_SIZE=args.legend_size, XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)
    
    plot_EC(HLFs, args.dataset_num, args.output_dir, args.particle_type, model_names, model_to_color_dict,'phi',
                ratio=False, row=args.row, col=args.col, height=4, width=6,
                YMAX=100, LEGEND_SIZE=args.legend_size, XLABEL_SIZE=args.xlabel_size, YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size, XTICK_SIZE=args.xtick_size, YTICK_SIZE=args.ytick_size)

     
def main():
    
    ### ......input arguments....
    args = parse_arguments()
    print("printing all arguments.....\n")
    print(args)
    
    default_output_dir = 'results'
    current_datetime = datetime.now()
    current_time = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    save_output_dir = default_output_dir + '_' + str(current_time)
 
    if os.path.isdir(default_output_dir):
        os.rename(default_output_dir, save_output_dir)
        os.mkdir(default_output_dir)
    else:
        os.mkdir(default_output_dir)
 
    if args.metrics == 'all':
         ## this returns incident energy, showers, HLFs object for each sample dataset and their order in the folder
        Es, Showers, HLFs, model_names, files = initialize_HLFs(args.dataset_path, args.particle_type, args.binning_file)
        model_to_color_dict = create_model_to_color_dict(model_names)
    
        if args.dataset_num == 2 or args.dataset_num == 3:
            
            evaluate_metrics_ds_2_3(Es, Showers, HLFs, model_names, files, args, model_to_color_dict)
            #computing fpd_kpd
            evaluate_fpd_kpd(Es, Showers, HLFs, model_names, files, args)
            
        elif args.dataset_num == 1:
            
            evaluate_metrics_ds_1(Es, Showers, HLFs, model_names, files, args, model_to_color_dict)
        else:
            print(f"Error in {args.dataset_num}.")
        
    elif args.metrics == 'fpd-kpd':
        
        Es, Showers, HLFs, model_names, files = initialize_HLFs(args.dataset_path, args.particle_type, args.binning_file)
        evaluate_fpd_kpd(Es, Showers, HLFs, model_names, files, args)
    elif args.metrics == 'CFD':
        Es, Showers, HLFs, model_names, files = initialize_HLFs(args.dataset_path, args.particle_type, args.binning_file)
        model_to_color_dict = create_model_to_color_dict(model_names)
        calc_cfd(Showers, model_names, args.output_dir, args.dataset_num, model_to_color_dict)
        
    elif args.metrics == 'layer':
        plot_layers(args)
        
    else:
        print(f"Error! {args.metrics} is not implemented.")
        

if __name__ == "__main__":
    main()
