
"""
This code is partially adopted from CaloChallenge github page. Here is the link for it
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
import math
import pandas as pd
from utils import write_dict_to_txt
from compute_metrics_helper import *
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def plot_sep_emd(filename, output_dir, dataset, particle, model_to_color_dict, width=7, height=5, taskname='separation_power'):
    """
    Plots separation power or EMD score for any number of models.
    """
    model_data = {}  # Dictionary to hold data for all models

    # Read and parse the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        key, value = line.strip().split(': ')
        value = float(value)

        # Extract model name before any underscore or digit
        model_name = key.split('_')[0]

        # Initialize model data if not present
        if model_name not in model_data:
            model_data[model_name] = {}

        # Add data
        index = sum(1 for _ in model_data[model_name])
        model_data[model_name][index] = math.log10(value)

    # Plotting
    fig, ax = plt.subplots(figsize=(width, height))

    markers = ['^', 'o', '*', 's', 'D', 'v', '<', '>']  # Marker styles for variety

    for idx, (model_name, data) in enumerate(model_data.items()):
        x = list(data.keys())
        y = list(data.values())
        marker = markers[idx % len(markers)]
        color = model_to_color_dict.get(model_name, plt.cm.tab10.colors[idx % 10])  # Use color from dict or default
        ax.plot(x, y, marker=marker, label=model_name, color=color)

    ax.set_xlabel('Layer number', fontsize=32)
    ax.set_ylabel(f'{taskname}', fontsize=28)

    pos = np.arange(len(next(iter(model_data.values()))))
    xtick_label = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44']

    ax.set_xticks(pos)
    ax.set_xticklabels(xtick_label)
    ax.set_ylim([-5, 0])
    ax.tick_params(axis='y', labelsize=28)
    ax.tick_params(axis='x', rotation=90, labelsize=28)

    ax.legend(loc='best', fontsize=20, borderpad=0.02, labelspacing=0.02,
              handlelength=0.4, handleheight=0.2, handletextpad=0.05,
              borderaxespad=0.05, columnspacing=0.05)

    fig.tight_layout()
    save_filename = f'{taskname}_dataset_{dataset}_particle_{particle}.pdf'
    save_path = os.path.join(output_dir, save_filename)
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")

    
def plot_cell_dist(shower_arr, min_energy, dataset, output_dir, particle, model_names, model_to_color_dict, width=8, height=5, TITLE_SIZE=30,
                   XLABEL_SIZE=25, YLABEL_SIZE=25, YMAX=1, ratio=False, LEGEND_SIZE=24, XTICK_SIZE=30, YTICK_SIZE=30):
    """ plots voxel energies across all layers """
    x_scale = 'log'
    EMDs = {}
    Seps = {}
    legend_names = ['Geant4']
    g_index=model_names.index('Geant4')
    ref_shower_arr = shower_arr[g_index] # trying to convert to GeV
    fig0, ax0 = plt.subplots(1, 1, figsize=(width*1, height*1), sharex=True, sharey=True)
    
    
    if x_scale == 'log':
        bins = np.logspace(np.log10(min_energy),
                           np.log10(ref_shower_arr.max()),
                           50)
    else:
        bins = 50

    eps = 1e-16
    counts_ref, _, _ = ax0.hist(ref_shower_arr.flatten(), bins=bins,
                                color = model_to_color_dict[model_names[g_index]],
                                label='Geant4', density=True, histtype='step',
                                alpha=1.0, linewidth=3.)
    
    for j in range(len(shower_arr)):
        if j==g_index:
            continue
        legend_names.append(model_names[j])
        counts_data, _, _ = ax0.hist(shower_arr[j].flatten() + eps, label=model_names[j], bins=bins, 
                                     color = model_to_color_dict[model_names[j]],
                                     histtype='step', linewidth=3., alpha=0.5, density=True)
        
        
        emd_score=get_emd(counts_ref,counts_data)
        EMDs[model_names[j]]=emd_score
        
        seps = separation_power(counts_ref, counts_data, bins)
        Seps[model_names[j]]=seps

        ax0.set_ylabel('A.U.', fontsize=YLABEL_SIZE)
        ax0.set_ylim([None, YMAX])
        #ax0.set_yscale('log')
        ax0.margins(0.05, 0.5)
        ax0.tick_params(axis='x', labelsize=XTICK_SIZE)
        ax0.tick_params(axis='y', labelsize=YTICK_SIZE)
        #ax0.set_ylabel("Arbitrary units")
        #plt.xlabel(r"Voxel Energy [MeV]")
        ax0.set_yscale('log')
        if x_scale == 'log': ax0.set_xscale('log')
    #plt_label = "Voxel Energy Distribution for Dataset "+dataset
    #fig0.suptitle(plt_label,y=1.1,fontsize=TITLE_SIZE) # for positioning title
    fig0.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig0.legend(lines, labels, loc='upper center',ncol = 4,fontsize=LEGEND_SIZE,bbox_to_anchor=[0.55, 1.02]) #for positioning figure
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=[0.5, 1.06], ncol=4,
               borderpad=0.1, labelspacing=0.1, handlelength=1.0, handleheight=0.5,
               handletextpad=0.2, borderaxespad=0.2, columnspacing=0.2)

    filename0 = os.path.join(output_dir, 'E_voxel_dataset_{}_particle_{}.pdf'.format( str(dataset), particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    emd_file = os.path.join(output_dir, "emd_E_voxel_dataset_{}_particle_{}.txt".format(str(dataset), particle))
    write_dict_to_txt(EMDs, emd_file)
    sep_file = os.path.join(output_dir, "separation_E_voxel_dataset_{}_particle_{}.txt".format(str(dataset), particle))
    write_dict_to_txt(Seps, sep_file)
    plt.close()

    
def plot_Etot_Einc_new(HLFs, dataset, output_dir, particle, model_names, model_to_color_dict, ratio=False, row=1, col=1, height=6, width=8, YMAX=100,
                   LEGEND_SIZE=24, XLABEL_SIZE=36, YLABEL_SIZE=36, TITLE_SIZE=48, XTICK_SIZE=30, YTICK_SIZE=30):
    """ plots Etot normalized to Einc histogram """
    EMDs = {}
    Seps = {}
    fig0, ax0 = plt.subplots(row, col, figsize=(width*col, height*row), sharex=True, sharey=True)
    # if("pion" in dataset):
    #     xmin, xmax = (0., 2.0)
    # else:
    legend_names = ['Geant4']
    xmin, xmax = (0.5, 1.5)
    g_index = model_names.index('Geant4')
    reference_class = HLFs[g_index]
    bins = np.linspace(xmin, xmax, 101)
  

    counts_ref, _, _ = ax0.hist(reference_class.GetEtot() / reference_class.Einc.squeeze(),
                                bins=bins, label='Geant4', density=True, color = model_to_color_dict[model_names[g_index]],
                                histtype='step', alpha=1.0, linewidth=3.)
    
    for j in range(len(HLFs)):
        if j == g_index:
            continue
        legend_names.append(model_names[j])
        counts_data, _, _ = ax0.hist(HLFs[j].GetEtot() / HLFs[j].Einc.squeeze(), bins=bins,color =model_to_color_dict[model_names[j]] ,
                                     label=model_names[j], histtype='step', linewidth=3., alpha=0.8,
                                     density=True)
        emd_score=get_emd(counts_ref, counts_data)
        EMDs[model_names[j]] = emd_score
        
        seps=separation_power(counts_ref,counts_data,bins)
        Seps[model_names[j]]=seps

        ax0.set_ylabel('A.U.', fontsize=YLABEL_SIZE)
        #ax0.set_xlabel(r"$E_{total}/E_{inc}$",fontsize=XLABEL_SIZE)

        ax0.set_ylim([None, YMAX])
        #ax0.set_yscale('log')
        ax0.margins(0.05, 0.5) 
        ax0.tick_params(axis='x', labelsize=XTICK_SIZE)
        ax0.tick_params(axis='y', labelsize=YTICK_SIZE)
    plt_label = "$E_{ratio}$ for Dataset " + str(dataset)
    #fig0.suptitle(plt_label,y=1.1,fontsize=TITLE_SIZE) # for positioning title
    fig0.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig0.legend(legend_names[:3], loc='upper center',ncol = 4,fontsize=LEGEND_SIZE,bbox_to_anchor=[0.5, 1.02]) #for positioning figure
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=[0.5, 1.06], ncol=4,
               borderpad=0.1, labelspacing=0.1, handlelength=1.0, handleheight=0.5,
               handletextpad=0.2, borderaxespad=0.2, columnspacing=0.2)
    filename0 = os.path.join(output_dir, 'E_ratio_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')


    emd_file = os.path.join(output_dir, "emd_E_ratio_dataset_{}_particle_{}.txt".format(str(dataset), particle))
    write_dict_to_txt(EMDs, emd_file)
    sep_file = os.path.join(output_dir, "separation_E_ratio_dataset_{}_particle_{}.txt".format(str(dataset), particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()

            
def plot_sparsity_group(list_hlfs, dataset,output_dir, particle, model_names, model_to_color_dict, ratio=False, row=3, col=3, height=6, width=8, YMAX=100,
                   LEGEND_SIZE=24, XLABEL_SIZE=36, YLABEL_SIZE=36, TITLE_SIZE=48, XTICK_SIZE=30, YTICK_SIZE=30):
    fig0, ax0 = plt.subplots(row, col, figsize=(width*col, height*row), sharex=True, sharey=False)
    """
        generates plots of sparsity distribution for dataset 2 and 3.
    """
    EMDs = {}
    Seps = {}
    gkeys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    dataset = str(dataset)
    legend_names = ['Geant4']
    for out_idx,keys in enumerate(gkeys):
       
        if dataset in ['2', '3']:
            lim = (0, 1.0)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        
        g_index = model_names.index('Geant4')
        
        reference_class = list_hlfs[g_index]
        
        shape_a = reference_class.GetSparsity()[0].shape[0]

        selected_ref = [(1-reference_class.GetSparsity()[i]).reshape(shape_a, 1) for i in keys]#turning into GeV
        combined_ref = np.concatenate(selected_ref, axis=1)

        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
        mean_ref = mean_ref.flatten()
        
        main_label = model_names[g_index] if out_idx==0 else None
        
        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, bins=bins,color = model_to_color_dict[model_names[g_index]],
                                    label=main_label, density=True, histtype='step',
                                    alpha=1.0, linewidth=3.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                legend_names.append(model_names[i])
                
                shape_a = reference_class.GetSparsity()[0].shape[0]

                selected_ref = [(1-list_hlfs[i].GetSparsity()[j]).reshape(shape_a, 1) for j in keys]#turning into GeV
                combined_ref = np.concatenate(selected_ref, axis=1)

                mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
                mean_ref = mean_ref.flatten()
                
                sub_label = model_names[i] if out_idx==0 else None
        
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, label=sub_label, bins=bins, color=model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=2., alpha=0.8, density=True)
            
                emd_score = get_emd(counts_ref, counts_data)
                EMDs[model_names[i] + "_" + str(keys[0]) + " to " + str(keys[4])] = emd_score

                seps = separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i] + "_" + str(keys[0]) + " to " + str(keys[4])] = seps
                
            cur_ylabel = "A.U." if out_idx%col==0 else ''
            ax0[out_idx//col][out_idx%col].set_ylabel(cur_ylabel, fontsize=YLABEL_SIZE)
            ax0[out_idx//col][out_idx%col].set_xlabel(r"Layer {} - {}".format(keys[0], keys[4]), fontsize=XLABEL_SIZE)
            #plt.xlabel(r'[mm]')
            #ax0[out_idx//col][out_idx%col].set_xlim(*lim)
            #ax0[out_idx//col][out_idx%col].set_ylim([0,YMAX])
            ax0[out_idx//col][out_idx%col].set_yscale('log')
            ax0[out_idx//col][out_idx%col].margins(0.05, 0.5) 
            ax0[out_idx//col][out_idx%col].tick_params(axis='x', labelsize=XTICK_SIZE)
            ax0[out_idx//col][out_idx%col].tick_params(axis='y', labelsize=YTICK_SIZE)
    
        
    plt_label = "Sparsity for Dataset "+str(dataset)
    #fig0.suptitle(plt_label,y=1.1,fontsize=TITLE_SIZE) # for positioning title
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'Sparsity_dataset_{}_particle_{}.pdf'.format( str(dataset),particle))
   
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig0.legend(legend_names[:3], loc='upper center', bbox_to_anchor=[0.55, 1.06], ncol=4, fontsize=LEGEND_SIZE,
               borderpad=0.1, labelspacing=0.1, handlelength=1.0, handleheight=0.5,
               handletextpad=0.2, borderaxespad=0.2, columnspacing=0.2) #for positioning figure
    
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    emd_file = os.path.join(output_dir, "emd_sparsity_dataset_{}_particle_{}.txt".format(str(dataset), particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file = os.path.join(output_dir, "separation_sparsity_dataset_{}.txt".format(str(dataset), particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    taskname = 'separation_power_sparsity'
    plot_sep_emd(sep_file, output_dir, dataset, particle, model_to_color_dict, width=7, height=5, taskname=taskname)
    taskname = 'emd_score_sparsity'
    plot_sep_emd(emd_file, output_dir, dataset, particle, model_to_color_dict, width=7, height=5, taskname=taskname)


def configure_subplot(ax, out_idx, key=None, keys=None, YLABEL_SIZE=12, XLABEL_SIZE=12, 
                      YMAX=None, XTICK_SIZE=10, YTICK_SIZE=10, col=None, row=None):
    """
    Configures a subplot with appropriate labels, scale, and appearance.
    """
    if keys:
        xlabel = f"Layer {keys[0]} - {keys[4]}"  # Use range format
    elif key is not None:
        xlabel = f"Layer {key}"  # Use single key format
    else:
        raise ValueError("Either 'keys' or 'key' must be provided.")

    if row > 1:  # Handling 2D subplot grid
        row_idx, col_idx = divmod(out_idx, col)  # Convert linear index to 2D
        subplot_ax = ax[row_idx, col_idx]  # Correctly index 2D array
        cur_ylabel = "A.U." if col_idx == 0 else '' 
    else:  # Handling 1D subplot list
        subplot_ax = ax.flatten()[out_idx]  # Flatten to ensure correct access
        cur_ylabel = "A.U." if out_idx == 0 else ''

    # Set labels and formatting
    subplot_ax.set_ylabel(cur_ylabel, fontsize=YLABEL_SIZE)
    subplot_ax.set_xlabel(xlabel, fontsize=XLABEL_SIZE)
    subplot_ax.set_ylim([None, YMAX])
    subplot_ax.set_yscale('log')
    subplot_ax.margins(0.05, 0.5)
    subplot_ax.tick_params(axis='x', labelsize=XTICK_SIZE)
    subplot_ax.tick_params(axis='y', labelsize=YTICK_SIZE)

    

def plot_EC_group(list_hlfs, dataset, output_dir, particle, model_names,model_to_color_dict, mode='eta', ratio=False, row=3, col=3, height=6, width=8, YMAX=100,
                   LEGEND_SIZE=24, XLABEL_SIZE=36, YLABEL_SIZE=36, TITLE_SIZE=48, XTICK_SIZE=30, YTICK_SIZE=30):
    """Plots center of energy in eta or phi for the given dataset."""
  

    get_EC = lambda obj: obj.GetECEtas() if mode == 'eta' else obj.GetECPhis()
    fig0, ax0 = plt.subplots(row, col, figsize=(width * col, height * row), sharex=True, sharey=True)
    
    
    EMDs, Seps = {}, {}
    legend_names = ['Geant4']
    gkeys = [[i + j for j in range(5)] for i in range(0, 45, 5)]
    
    for out_idx, keys in enumerate(gkeys):
        lim = (-45., 45.) if dataset in ['2', '3'] else (-100., 100.)
        bins = np.linspace(*lim, 101)
        g_index = model_names.index('Geant4')
        reference_class = list_hlfs[g_index]

        shape_a = get_EC(reference_class)[0].shape[0]
        selected_ref = [get_EC(reference_class)[i].reshape(shape_a, 1) for i in keys]
        combined_ref = np.concatenate(selected_ref, axis=1)
        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
        
        main_label = model_names[g_index] if out_idx == 0 else None
        counts_ref, _, _ = ax0[out_idx // col][out_idx % col].hist(mean_ref, bins=bins, color=model_to_color_dict[model_names[g_index]],
                                                                   label=main_label, density=True, histtype='step',
                                                                   alpha=1.0, linewidth=3.)

        for i, hlf in enumerate(list_hlfs):
            if hlf is None or g_index == i:
                continue

            legend_names.append(model_names[i])
            shape_a = get_EC(hlf)[0].shape[0]
            selected_data = [get_EC(hlf)[j].reshape(shape_a, 1) for j in keys]
            combined_data = np.concatenate(selected_data, axis=1)
            mean_data = np.mean(combined_data, axis=1, keepdims=True)

            sub_label = model_names[i] if out_idx == 0 else None
            counts_data, _, _ = ax0[out_idx // col][out_idx % col].hist(mean_data, label=sub_label, bins=bins,
                                                                        color=model_to_color_dict[model_names[i]],
                                                                        histtype='step', linewidth=3., alpha=0.8, density=True)
            
            emd_score = get_emd(counts_ref, counts_data)
            EMDs[f"{model_names[i]}_{keys[0]} to {keys[4]}"] = emd_score

            seps = separation_power(counts_ref, counts_data, bins)
            Seps[f"{model_names[i]}_{keys[0]} to {keys[4]}"] = seps
        
        configure_subplot(ax0, out_idx, None, keys, YLABEL_SIZE, XLABEL_SIZE, 
                      YMAX, XTICK_SIZE, YTICK_SIZE, col,row)

    fig0.tight_layout()
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=[0.5, 1.06],
                ncol=4, borderpad=0.1, labelspacing=0.1, handlelength=1.0, handleheight=0.5,
                handletextpad=0.2, borderaxespad=0.2, columnspacing=0.2)

    filename0 = os.path.join(output_dir, f'EC_{mode}_dataset_{dataset}_particle_{particle}.pdf')
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')

    

    emd_file = os.path.join(output_dir, f"emd_EC_{mode}_dataset_{dataset}_particle_{particle}.txt")
    write_dict_to_txt(EMDs, emd_file)
    sep_file = os.path.join(output_dir, f"separation_EC_{mode}_dataset_{dataset}_particle_{particle}.txt")
    write_dict_to_txt(Seps, sep_file)

    plot_sep_emd(sep_file, output_dir, dataset, particle,model_to_color_dict, width=7, height=5, taskname=f'separation_power_EC_{mode}')
    plot_sep_emd(emd_file, output_dir, dataset, particle,model_to_color_dict, width=7, height=5, taskname=f'emd_score_EC_{mode}')
    plt.close()

    
def plot_SW_group(list_hlfs, dataset,output_dir, particle,model_names, model_to_color_dict,mode='eta',                  ratio=False,row=3,col=3,height=6,width=8,YMAX=100,
                   LEGEND_SIZE=24,XLABEL_SIZE=36,YLABEL_SIZE=36,TITLE_SIZE=48,XTICK_SIZE=30,YTICK_SIZE=30):
    
    """ plots shower width in phi direction for dataset 2 and 3 """
    get_SW = lambda obj: obj.GetWidthEtas() if mode == 'eta' else obj.GetWidthPhis()
    fig0, ax0 = plt.subplots(row,col,figsize=(width*col,height*row),sharex=True,sharey=True)

    EMDs={}
    Seps={}
    legend_names=['Geant4']
    gkeys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    dataset=str(dataset)
    
    for out_idx,keys in enumerate(gkeys):
        
        if dataset in ['2', '3']:
            lim = (0, 30.)
        else:
            lim = (-100., 100.)

        bins = np.linspace(*lim, 101)
        
        g_index=model_names.index('Geant4')
       
        reference_class=list_hlfs[g_index]

        shape_a=get_SW(reference_class)[0].shape[0]

        selected_ref = [get_SW(reference_class)[i].reshape(shape_a,1) for i in keys]#turning into GeV
        combined_ref = np.concatenate(selected_ref, axis=1)
        mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
        main_label = model_names[g_index] if out_idx==0 else None

        counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, bins=bins,color = model_to_color_dict[model_names[g_index]],
                                    label=main_label, density=True, histtype='step',
                                    alpha=1.0, linewidth=3.)
        
        for i, hlf in enumerate(list_hlfs):
            if list_hlfs[i] == None or g_index==i:
                pass
            else:
                legend_names.append(model_names[i])
               
                shape_a=get_SW(hlf)[0].shape[0]

                selected_ref = [get_SW(hlf)[j].reshape(shape_a,1) for j in keys]#turning into GeV
                combined_ref = np.concatenate(selected_ref, axis=1)

                mean_ref = np.mean(combined_ref, axis=1, keepdims=True)
                sub_label = model_names[i] if out_idx==0 else None
        
                counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(mean_ref, label=sub_label, bins=bins, color =  model_to_color_dict[model_names[i]],
                                             histtype='step', linewidth=2., alpha=0.8, density=True)
            
                emd_score=get_emd(counts_ref,counts_data)
                EMDs[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=emd_score

                seps = separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i]+"_"+str(keys[0])+" to "+str(keys[4])]=seps

            configure_subplot(ax0, out_idx, None, keys, YLABEL_SIZE, XLABEL_SIZE, 
                      YMAX, XTICK_SIZE, YTICK_SIZE, col,row)
            
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig0.legend(lines, labels, loc='upper center',ncol = 2,fontsize=LEGEND_SIZE,bbox_to_anchor=[0.5, 1.18]) #for positioning figure
    fig0.legend(legend_names[:len(model_names)], fontsize=LEGEND_SIZE,loc='upper center', bbox_to_anchor=[0.5, 1.06],ncol=4,
               borderpad=0.1,labelspacing=0.1,handlelength=1.0,handleheight=0.5,
               handletextpad=0.2,borderaxespad=0.2,columnspacing=0.2)
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, 'SW_{}_dataset_{}_particle_{}.pdf'.format(mode, dataset,particle))
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')
    
    
    emd_file=os.path.join(output_dir,"emd_SW_{}_dataset_{}_particle_{}.txt".format(mode,dataset,particle))
    write_dict_to_txt(EMDs,emd_file)
    sep_file=os.path.join(output_dir,"separation_SW_{}_dataset_{}_particle_{}.txt".format(mode,dataset,particle))
    write_dict_to_txt(Seps,sep_file)
    plt.close()
    taskname=f'separation_power_SW_{mode}'
    plot_sep_emd(sep_file, output_dir, dataset, particle,model_to_color_dict,width=7,height=5,taskname=taskname)
    taskname=f'emd_score_SW_{mode}'
    plot_sep_emd(emd_file, output_dir, dataset, particle,model_to_color_dict,width=7,height=5,taskname=taskname)
    
    
def plot_SW(list_hlfs, dataset, output_dir, particle, model_names, model_to_color_dict, direction='eta', ratio=False, row=1, col=2, height=6, width=8, YMAX=100,
            LEGEND_SIZE=24, XLABEL_SIZE=36, YLABEL_SIZE=36, TITLE_SIZE=48, XTICK_SIZE=30, YTICK_SIZE=30):
    """ plots shower width in eta or phi direction for the given dataset """
    EMDs = {}
    Seps = {}
    
    fig0, ax0 = plt.subplots(row, col, figsize=(width*col, height*row), sharex=True, sharey=True)
    # if row>1:
    #     ax0=ax0.flatten()
        
    
    g_index = model_names.index('Geant4')
    reference_class = list_hlfs[g_index]
    
    if direction == 'eta':
        width_dict = reference_class.GetWidthEtas()
        label = "$\\eta$ (mm) direction"
        get_width = lambda x: x.GetWidthEtas()
    elif direction == 'phi':
        width_dict = reference_class.GetWidthPhis()
        label = "$\\phi$ (mm) direction"
        get_width = lambda x: x.GetWidthPhis()
    else:
        raise ValueError("Invalid direction. Use 'eta' or 'phi'.")
    
    for out_idx, key in enumerate(width_dict.keys()):
        if dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (0., 500.)
        else:
            lim = (0., 300.)

        bins = np.linspace(*lim, 101)
        if particle=='photon':
            counts_ref, _, _ = ax0[out_idx].hist(get_width(reference_class)[key], bins=bins,
                                             color=model_to_color_dict[model_names[g_index]],
                                             label=model_names[g_index], density=True, histtype='step',
                                             alpha=1.0, linewidth=2.)
        else:
            counts_ref, _, _ = ax0[out_idx//col][out_idx%col].hist(get_width(reference_class)[key], bins=bins,
                                                 color=model_to_color_dict[model_names[g_index]],
                                                 label=model_names[g_index], density=True, histtype='step',
                                                 alpha=1.0, linewidth=2.)
        
        for i in range(len(list_hlfs)):
            if list_hlfs[i] is None or g_index == i:
                pass
            else:
                if particle=='photon':
                    counts_data, _, _ = ax0[out_idx].hist(get_width(list_hlfs[i])[key], label=model_names[i], bins=bins,
                                                      color=model_to_color_dict[model_names[i]],
                                                      histtype='step', linewidth=3., alpha=0.5, density=True)
                else:
                    counts_data, _, _ = ax0[out_idx//col][out_idx%col].hist(get_width(list_hlfs[i])[key], label=model_names[i], bins=bins,
                                                          color=model_to_color_dict[model_names[i]],
                                                          histtype='step', linewidth=3., alpha=0.5, density=True)

                emd_score = get_emd(counts_ref, counts_data)
                EMDs[model_names[i] + "_" + str(key)] = emd_score

                seps = separation_power(counts_ref, counts_data, bins)
                Seps[model_names[i] + "_" + str(key)] = seps

            configure_subplot(ax0, out_idx, key, None, YLABEL_SIZE, XLABEL_SIZE, 
                      YMAX, XTICK_SIZE, YTICK_SIZE, col,row)

    fig0.suptitle(f"Shower width in {label} for Dataset {dataset}", y=1.10, fontsize=TITLE_SIZE)
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig0.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig0.legend(lines[:4], labels[:4], loc='upper center', bbox_to_anchor=[0.55, 1.08],
                ncol=4, fontsize=LEGEND_SIZE, borderpad=0.1, labelspacing=0.1, handlelength=1.0, handleheight=0.5,
                handletextpad=0.2, borderaxespad=0.2, columnspacing=0.2)
    
    fig0.tight_layout()
    filename0 = os.path.join(output_dir, f'SW_{direction}_dataset_{dataset}_particle_{particle}.pdf')
    fig0.savefig(filename0, dpi=350, bbox_inches='tight')

    emd_file = os.path.join(output_dir, f"emd_SW_{direction}_dataset_{dataset}_particle_{particle}.txt")
    write_dict_to_txt(EMDs, emd_file)
    
    sep_file = os.path.join(output_dir, f"separation_SW_{direction}_dataset_{dataset}_particle_{particle}.txt")
    write_dict_to_txt(Seps, sep_file)
    
    plt.close()



def plot_EC(list_hlfs, dataset, output_dir, particle, model_names,model_to_color_dict, direction='eta', ratio=False,
            row=2, col=2, height=6, width=8, YMAX=100,
            LEGEND_SIZE=24, XLABEL_SIZE=36, YLABEL_SIZE=36, TITLE_SIZE=48, XTICK_SIZE=30, YTICK_SIZE=30):
    """Plots center of energy in eta or phi for the given dataset."""
    EMDs, Seps = {}, {}
    fig, ax = plt.subplots(row, col, figsize=(width * col, height * row), sharex=False, sharey=True, squeeze=False)
    
    g_index = model_names.index('Geant4')
    dataset = str(dataset)
    reference_class = list_hlfs[g_index]
    
    get_EC = reference_class.GetECEtas if direction == 'eta' else reference_class.GetECPhis
    lim_map = {12: (-300., 300.), 13: (-300., 300.)} if direction == 'eta' else {12: (-250., 250.), 13: (-250., 250.)}
    
    for out_idx, key in enumerate(get_EC().keys()):
        lim = lim_map.get(key, (-30., 30.) if dataset in ['2', '3'] else (-100., 100.))
        bins = np.linspace(*lim, 101)

        counts_ref, _, _ = ax[out_idx // col][out_idx % col].hist(
            get_EC()[key], bins=bins, color=model_to_color_dict[model_names[g_index]],
            label=model_names[g_index], density=True, histtype='step', alpha=1.0, linewidth=2.)

        for i in range(len(list_hlfs)):
            if list_hlfs[i] is None or g_index == i:
                continue

            get_EC_data = list_hlfs[i].GetECEtas if direction == 'eta' else list_hlfs[i].GetECPhis
            counts_data, _, _ = ax[out_idx // col][out_idx % col].hist(
                get_EC_data()[key], label=model_names[i], bins=bins,
                color=model_to_color_dict[model_names[i]], histtype='step', linewidth=3., alpha=0.5, density=True)

            EMDs[f"{model_names[i]}_{key}"] = get_emd(counts_ref, counts_data)
            Seps[f"{model_names[i]}_{key}"] = separation_power(counts_ref, counts_data, bins)
        
        configure_subplot(ax, out_idx, key, None, YLABEL_SIZE, XLABEL_SIZE, 
                      YMAX, XTICK_SIZE, YTICK_SIZE, col,row)
    
    lines_labels = [ax_.get_legend_handles_labels() for ax_ in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:len(model_names)], labels[:len(model_names)], loc='upper center', bbox_to_anchor=[0.55, 1.08],
               ncol=4, fontsize=LEGEND_SIZE, borderpad=0.1, labelspacing=0.1, handlelength=1.0, handleheight=0.5,
               handletextpad=0.2, borderaxespad=0.2, columnspacing=0.2)
    fig.tight_layout()
    
    filename = os.path.join(output_dir, f'EC_{direction}_dataset_{dataset}_particle_{particle}.pdf')
    fig.savefig(filename, dpi=350, bbox_inches='tight')
    
    write_dict_to_txt(EMDs, os.path.join(output_dir, f'emd_EC_{direction}_dataset_{dataset}_particle_{particle}.txt'))
    write_dict_to_txt(Seps, os.path.join(output_dir, f'separation_EC_{direction}_dataset_{dataset}_particle_{particle}.txt'))
    plt.close()
    
    
def set_x_ticks_and_labels(ax, xy, dataset):
    if xy == 'a':
        ticks = np.arange(16) if dataset != '3' else np.arange(0, 50, 5)
        labels = np.arange(1, 17) if dataset != '3' else np.arange(1, 51, 5)
        xlabel = 'Angular-bins'
    elif xy == 'r':
        ticks = np.arange(9) if dataset != '3' else np.arange(18)
        labels = np.arange(1, 10) if dataset != '3' else np.arange(1, 19)
        xlabel = 'Radial-bins'
    elif xy == 'z':
        ticks = np.arange(0, 45, 5)
        labels = np.arange(1, 46, 5)
        xlabel = 'Layers'
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=14)  # Increase tick label size
    ax.set_xlabel(xlabel, fontsize=16)  # Increase x-axis label size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick size

def plot_data_with_styles(ax, data, labels, markers, ls, xy, dataset,model_to_color_dict):
    # Plotting data with markers and line styles
    for i in range(len(data)):
        color = model_to_color_dict.get(labels[i], 'black') 
        ax.plot(data[i], marker=markers[i], linestyle=ls, label=labels[i],color=color)

    # Setting x-ticks and labels
    set_x_ticks_and_labels(ax, xy, dataset)

    ax.set_ylabel(f'Mean Energy [GeV] in {xy}', fontsize=16)  # Increase y-axis label size
    ax.legend(fontsize=14)  # Increase legend font size
    ax.set_title(f'Mean Energy - {xy} Bins', fontsize=18)  # Increase title font size
    
def draw_voxel_heatmap(corr_mat_gen, corr_mat_ref, name, output_dir,model_name):
    """
    Generates a PDF with side-by-side correlation heatmaps for two datasets.
    
    Parameters:
        corr_mat_gen (list of ndarray): Correlation matrices for the generated data.
        corr_mat_ref (list of ndarray): Correlation matrices for the reference data.
        output_pdf (str): Output file name for the PDF.
        
    """
   
    if len(corr_mat_gen) != len(corr_mat_ref):
        raise ValueError("The number of layers in generated and reference data must match.")
    output_pdf = os.path.join(output_dir, name)

    # Create a PDF object to store plots
    with PdfPages(output_pdf) as pdf:
        fig_title = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5,model_name, ha='center', va='center', fontsize=20, fontweight='bold')
        plt.axis('off')  # Hide axis
        pdf.savefig(fig_title)  # Save the title page
        plt.close(fig_title)
        for idx, (gen_mat, ref_mat) in enumerate(zip(corr_mat_gen, corr_mat_ref)):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Plot for generated data
            sns.heatmap(gen_mat, ax=axes[0], cmap="coolwarm", annot=False, cbar=True,vmin=0,vmax=1)
            axes[0].set_title(f"{model_name} Data: Layer {idx+1} and {idx+2}")
            axes[0].set_xlabel("Voxel in radial bins")
            axes[0].set_ylabel("Voxel in angular bins")

            # Plot for reference data
            sns.heatmap(ref_mat, ax=axes[1], cmap="coolwarm", annot=False, cbar=True,vmin=0,vmax=1)
            axes[1].set_title(f"Geant4 Data: Layer {idx+1} and {idx+2}")
            axes[1].set_xlabel("Voxel in radial bins")
            axes[1].set_ylabel("Voxel in angular bins")

            # Adjust layout and save to PDF
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
def draw_heatmap(correlation_matrix,name,output_dir,width=8,height=6,TITLE_SIZE=30
                   ,XLABEL_SIZE=25,YLABEL_SIZE=25,LEGEND_SIZE=16,XTICK_SIZE=24,YTICK_SIZE=24,STEPSIZE=5,CBAR=True):
    
    """
    This is a simple function to fraw the heatmap for layer wise correlation
    """
    sns.set()  # Set seaborn style
    fig,ax = plt.subplots(figsize=(width, height))  # Set the figure size
    
    
    sns.heatmap(correlation_matrix, ax=ax,annot=False, cmap='coolwarm', fmt='.2f',cbar=CBAR,vmin=0,vmax=1)
    ax.set_xlabel('Layers',fontsize=XLABEL_SIZE)
    ax.set_ylabel('Layers',fontsize=YLABEL_SIZE)
    ax.tick_params(axis='x', rotation=90,labelsize=XTICK_SIZE)
    ax.tick_params(axis='y', rotation=0,labelsize=YTICK_SIZE)
   
    if CBAR:
        cbar = ax.collections[0].colorbar
        # here set the labelsize by 20
        cbar.ax.tick_params(labelsize=LEGEND_SIZE)
    plt.gca().invert_yaxis()
    
    fig.savefig(name, bbox_inches='tight',dpi=350)  # Save the figure
    # Show the heatmap
    save_path = os.path.join(output_dir, name)
    plt.savefig(save_path)

def plot_mean_energy(Showers, model_names, model_to_color_dict, dataset, output_dir, particle):
    angular_bins, radial_bins, z_bins = [], [], []
    
    # Dataset-specific settings
   
    if dataset == 2:
        a, r, l = 16, 9, 45
        shape=[-1,45,16,9]
    elif dataset == 3:
        a, r, l = 50, 18, 45
        shape=[-1,45,50,18]
    else:
        print("Not implemented yet!")
        return
    
    # Get mean energy for each model
    for i, m in enumerate(model_names):
        s=Showers[i].reshape(shape)
        angular_bins.append(mean_energy('a', s).reshape([a, 1]).flatten())
        radial_bins.append(mean_energy('r', s).reshape([r, 1]).flatten())
        z_bins.append(mean_energy('z', s).reshape([l, 1]).flatten())
    
    markers = ['^', 'o', '*', 's', 'D', 'v', '<', '>']  # Marker styles
    ls = '-'  # Line style
    
    # Plot and save the angular bins plot
    fig_a, ax_a = plt.subplots(figsize=(10, 6))
    plot_data_with_styles(ax_a, angular_bins, model_names, markers, ls, 'a', dataset,model_to_color_dict)
    fig_a.savefig(f'{output_dir}/mean_energy_angular_bins_{dataset}_{particle}.pdf')
    
    # Plot and save the radial bins plot
    fig_r, ax_r = plt.subplots(figsize=(10, 6))
    plot_data_with_styles(ax_r, radial_bins, model_names, markers, ls, 'r', dataset,model_to_color_dict)
    fig_r.savefig(f'{output_dir}/mean_energy_radial_bins_{dataset}_{particle}.pdf')
    
    # Plot and save the z bins plot
    fig_z, ax_z = plt.subplots(figsize=(10, 6))
    plot_data_with_styles(ax_z, z_bins, model_names, markers, ls, 'z', dataset,model_to_color_dict)
    fig_z.savefig(f'{output_dir}/mean_energy_z_bins_{dataset}_{particle}.pdf')

    # No need to show the plots, so I removed the plt.show()
    plt.tight_layout()

    
def plot_frob_norm(frobs,name,model_names,output_dir,mode, model_to_color_dict):
    
    colors = [model_to_color_dict.get(model, 'skyblue') for model in model_names]
    # Plotting frobenius norm as a bar_plot
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, frobs, color=colors, edgecolor='black')
    
    # Customize the plot
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Frobenius Norm with Geant4", fontsize=12)
    plt.title(f"Frobenius Norm Comparison with Geant4 for {mode} wise correlation", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'frob_norm_{name}')
    plt.savefig(save_path)
   
    
def draw_plots_cfd(correlations,g_idx,name,model_names,out_dir,mode,model_to_color_dict):
    frobs=[]
    re_models=[]
    corr_g=correlations[g_idx]
    

    #if plot=='heatmap':
    
    for i,corr in enumerate(correlations):
        file_name='heatmap_'+model_names[i]+'_'+name
        if mode=='voxel':
            draw_voxel_heatmap(corr,correlations[g_idx],file_name,out_dir,model_names[i])
        elif mode=='layer':
            draw_heatmap(corr,file_name,output_dir=out_dir)
        else: 
            print("not implemented heatmap for grouped voxel")
                
            
    #elif plot=='bar_plot':
    for i,corr in enumerate(correlations):
        if i!=g_idx:
            f=calculate_frob_norm(corr_g, corr)
            frobs.append(f)
            re_models.append(model_names[i])
    plot_frob_norm(frobs,name, re_models,out_dir,mode,model_to_color_dict)

    # else:
    #     print("will be updated later!")
def calc_cfd(Showers, model_names, out_dir,dataset,model_to_color_dict):
    
    g_idx=model_names.index('Geant4')
    if dataset==2:
        shape=[-1,45,16,9]
    elif dataset==3:
        shape=[-1,45,50,18]
    else:
        print('Not implemented yet for dataset 1 photon and pion')
        return
    
    mode='voxel'
    ## looking at the correlation between the voxel j of layer i and the voxel j of layer i+1
    correlations=[]

    for S in Showers:
        correlations.append(calculate_correlation_voxel(S.reshape(shape)))
    fileName=str(dataset)+'_'+mode+'.pdf'
    draw_plots_cfd(correlations,g_idx,fileName,model_names,out_dir,mode,model_to_color_dict)

    mode='layer'
    ## looking at the correlation between the layer i and layer i+1, considering their layer_sum
    correlations=[]
    for S in Showers:
        summ=S.reshape(shape).sum(axis=(2,3))
        corr,_=calculate_correlation_layer(summ)
        correlations.append(corr)
    fileName=str(dataset)+'_'+mode+'.pdf'    
    draw_plots_cfd(correlations,g_idx,fileName,model_names,out_dir,mode,model_to_color_dict)

    mode='group'
    ## First create a group of layers by combining 5 consecutive layers, then sum along the axis of angular bins  and
    ## finally compute correlation between group i's radial_bin j with group i+1's radial_bin j
    correlations=[]

    for S in Showers:
        data=grouping_data(S.reshape(shape))
        correlations.append(calculate_correlation_group(data))
    fileName=str(dataset)+'_'+mode+'.pdf'   
    draw_plots_cfd(correlations,g_idx,fileName,model_names,out_dir,mode,model_to_color_dict)

    
def plot_E_group_layers(ref_model, hlf_classes, model_names, plot_filename, e_range, dataset_num,args,model_to_color_dict):
    """ plots energy deposited in 5 consecutive layers by creating a group of 5 layers"""
    # this is only applicable for dataset 2 and dataset 3. Dataset 1 does not need this
    min_energy=0.5e-4/0.033
    x_scale='log'
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    keys = [[i+j for j in range(5)] for i in range(0, 45, 5)]
    axs = axs.flatten()

    print("Range: ", e_range)

    sep_powers_all = []
    layers_all = []
    EMDs={}
    Seps={}
    for i, key in enumerate(keys):
        
        sep_powers = []
        layers = []
        
        ref_shape = hlf_classes[ref_model].GetElayers()[0].shape[0]
        ref_selected = [hlf_classes[ref_model].GetElayers()[i].reshape(ref_shape, 1)/1000 for i in key]#turning into GeV
        ref_combined = np.concatenate(ref_selected, axis=1)
        #ref_mean = np.mean(ref_combined, axis=1, keepdims=True) 
        ref_mean = np.sum(ref_combined, axis=1, keepdims=True) 

        if x_scale == 'log':
            bins = np.logspace(np.log10(min_energy),
                               np.log10(ref_mean.max()), 40)
        else:
            bins = 40

        ref_counts, bins, _ = axs[i].hist(ref_mean, bins=bins, label=ref_model, density=True, 
                            histtype='step',color=model_to_color_dict[ref_model], alpha=0.2, linewidth=3.)

        this_layer = str(key[0]) + "-" + str(key[4]) 
        sep_powers.append(this_layer)
        sep_powers.append(0) # seperation power baseline, Geant4 with itself

        legend_names = ['Geant4']
            
        for j, model in enumerate(model_names):
            if model != ref_model:

                model_shape = hlf_classes[model].GetElayers()[0].shape[0]
                selected_hlf = [hlf_classes[model].GetElayers()[i].reshape(model_shape, 1)/1000 for i in key]#turning into GeV
                combined_hlf = np.concatenate(selected_hlf, axis=1)
              
                hlf_means = np.sum(combined_hlf, axis=1, keepdims=True) 
                model_counts, _, _ = axs[i].hist(hlf_means, label=model, bins=bins,
                                    histtype='step', color=model_to_color_dict[model], linewidth=3., alpha=1., density=True)
            
                axs[i].set_title("layer {} to {}".format(key[0],key[4]))
                axs[i].set_xlabel(r'$E$ [GeV]')
                axs[i].set_yscale('log')
                axs[i].set_xscale('log')
                emd_score=get_emd(ref_counts,model_counts)
                EMDs[model+"_"+str(key)]=emd_score

                seps = separation_power(ref_counts, model_counts, bins)
                Seps[model+"_"+str(key)]=seps

                legend_names.append(model)
           
        layers.append(str(key[0]) + "-" + str(key[4]))

      
    fig.legend(legend_names, fontsize=12, prop={'size': 10},
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncols=4)
    plt.tight_layout(pad=3.0)
    save_path = os.path.join(args.output_dir, plot_filename)
    plt.savefig(save_path, dpi=300)

    plt.close()
    
    
    new_path_emd=args.output_dir
    emd_file=os.path.join(new_path_emd,
                                    "emd_E_layer_dataset_{}_particle_{}.txt".format(dataset_num,args.particle_type))
    write_dict_to_txt(EMDs,emd_file)
    new_path_sep=args.output_dir
    sep_file=os.path.join(new_path_sep,
                              "separation_E_layer_dataset_{}_particle_{}.txt".format(dataset_num,args.particle_type))
    write_dict_to_txt(Seps,sep_file)
    
    
    taskname=f'separation_power_E_layer_{e_range}'
    plot_sep_emd(sep_file, args.output_dir, dataset_num, args.particle_type,model_to_color_dict,width=7,height=5,taskname=taskname)
    taskname=f'emd_score_E_layer_{e_range}'
    plot_sep_emd(emd_file, args.output_dir, dataset_num, args.particle_type,model_to_color_dict,width=7,height=5,taskname=taskname)
    


    
def plot_layers(args,model_to_color_dict,Showers, HLFs,Es,model_names):
  
    target_energies = 10**np.linspace(3, 6, 4)
    
    hlfs = dict()
    for model in model_names:
        hlfs[model] = HLF.HighLevelFeatures(args.particle_type, args.binning_file)

    # plot energy ranges seperated
    for i in range(len(target_energies)-1):
        for idx,model in enumerate(model_names):

            energies, showers = Es[idx],Showers[idx]
            
            
            shower_bins = ((energies >= target_energies[i]) & \
                             (energies < target_energies[i+1])).squeeze()
            hlfs[model].Einc = energies[shower_bins]
            hlfs[model].CalculateFeatures(showers[shower_bins])
            
        e_range = str(int(target_energies[i]/1000))+'GeV_'+str(int(target_energies[i+1]/1000))+'GeV'
        plot_filename =  'E_layers_dataset_{}_{}.pdf'.format(args.dataset_num, e_range)
        ref = 'Geant4'
        plot_E_group_layers(ref, hlfs, model_names, plot_filename, e_range, args.dataset_num,args,model_to_color_dict)

        


    # plot all energy ranges
    hlfs = dict()
    
    for idx,model in enumerate(model_names):
        hlfs[model]=HLFs[idx]
    plot_filename =  'E_layers_dataset_{}_{}.pdf'.format(args.dataset_num, 'all')
    ref = 'Geant4'
    e_range = 'all'
    plot_E_group_layers(ref, hlfs, model_names, plot_filename, e_range, args.dataset_num,args,model_to_color_dict)

    #print("done plotting...")
 
    
        
        
    
