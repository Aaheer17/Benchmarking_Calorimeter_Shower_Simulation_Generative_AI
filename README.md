# Benchmarking Calorimeter Shower Simulation using Generative AI

In this project we are comparing 4 different Generative AI models on Calorimeter Shower Simulation. This project is inspired by [CaloChallenge 2022](https://calochallenge.github.io/homepage/).

## Dataset

- Dataset 1 available at [Dataset 1](https://zenodo.org/records/8099322).
- Dataset 2 available at [Dataset 2](https://zenodo.org/records/6366271).
- Dataset 3 available at [Dataset 3](https://zenodo.org/records/6366324).
- You can access our generated samples here: [Dataset 2 Samples on Zenodo](https://zenodo.org/records/14883798).
We will upload our samples for dataset 1 and 3 soon. 

## Generative AI models

Here we compare 4 different Generative AI models.
They are 
1. CaloDream on dataset 2 and 3 (Conditional Flow Matching)https://github.com/luigifvr/calo_dreamer
2. CaloDiffusion (Denoising Diffusion based model) https://github.com/OzAmram/CaloDiffusionPaper/tree/main
3. CaloScore (Score based model) https://github.com/ViniciusMikuni/CaloScoreV2/tree/main
4. CaloINN (Combination of VAE and Normalizing Flow) https://github.com/heidelberg-hepml/CaloINN/tree/calochallenge

## Directory Structure

- The main evaluation scripts and helper modules to generate plots and various metrics are at the top level.
- The `xml_binning_files` folder contains the binning files in XML format needed to run the evaluation scripts.
- We will upload our trained models to Zenodo soon.
  
## List of Metrics
Below are the list of metrics to compare the performance of various models. For a detailed explanation of the metrics and how to compare, please 
refer to our paper https://arxiv.org/pdf/2406.12898

1. Histograms of physics observables
   - Layer wise energy distribution
   - Center of energy in η and φ direction
   - Shower width in η and φ direction
   - Sparsity
   - Mean energy distribution for angular, radial and z bins
3. Correlations
   - Pearson Correlation Coefficient
   - Frobenius Norm
5. Classifier tests
   - Area Under Curve (AUC)
   - Jensen-Shannon divergence (JSD)
7. Scores
   - Earth Mover’s Distance (EMD)
   - Fréchet Physics Distances(FPD)
   - Kernel Physics Distances(KPD)
9. Separation power
10. Training and Evaluation time
    
## Creating the environment for plotting
A suitable python environment named eval can be created and activated with:
```
python -m venv eval
source eval/bin/activate
pip install -r requirements.txt
```
## Evaluation Script Parameters
The following table lists the required parameters for running the evaluation script along with their usage:

| Parameters    | Usage |
|:------------:|:------|
| dataset_path | Path to the folder that contains samples from different models and Geant4. Files are saved in a pattern of `dataset_NUM_PARTICLE_MODEL.h5`. |
| metrics      | Options: `all`, `fpd-kpd`, `CFD`. |
| dataset_num  | Type of dataset: `1`, `2`, or `3`. |
| output_dir   | Path to the directory to save results. |
| binning_file | Path to the binning file. |
| particle_type | Type of the particle being evaluated, e.g., `photon`, `pion`, `electron`. |

The metric option `all` will cover all of the metrics described above.

The metric **CFD** stands for **Correlation Frobenius Distance**, which is one of our contributions. This metric measures how the consecutive layers and voxels of generated samples are correlated with each other compared to Geant4 samples.

CFD helps evaluate the consistency of energy deposition patterns across layers, capturing the spatial correlations in the calorimeter shower. Lower CFD values indicate that the generated samples better preserve the correlations observed in Geant4 simulations. Our current implemention is for dataset 2 and 3. It can easily be modified for dataset 1.

## Running the evaluation scripts

1. To generate Sparsity, Center of Energy, Shower width, voxel distribution, mean energy, and E_ratio plots, run the following commands:

```
# Dataset 1(photon)
python evaluate.py --metrics 'all' --binning_file 'xml_binning_files/file_name' --dataset_path 'path_to_dataset_path' --dataset_num 1 --particle_type 'photon' --row 1 --col 2

# Dataset 1 (pion):
python evaluate.py --metrics 'all' --binning_file 'xml_binning_files/file_name' --dataset_path 'path_to_dataset_path' --dataset_num 1 --particle_type 'pion' --row 2 --col 2

# Dataset 2 and 3:
python evaluate.py --metrics 'all' --binning_file 'xml_binning_files/file_name' --dataset_path 'path_to_dataset_path' --dataset_num '[2, 3]' --particle_type 'electron' --row 3 --col 3
```
2. To generate FPD and KPD scores, run the following commands:
```
python evaluate.py --metrics ‘fpd-kpd’ --binning_file ‘path_to_binning_file’ --dataset_path ‘path_to_dataset_path’ --dataset_num 'dataset_num' --particle_type ‘electron’ 
```

3. To generate correlation plots , run the following command:
```
python evaluate.py --metrics ‘CFD’ --binning_file ‘path_to_binning_file’ --dataset_path ‘path_to_dataset_path’ --dataset_num 'dataset_num' --particle_type ‘electron’ 
```

4. To generate AUC and JSD scores, run the following command:
```
python classifier_auc_jsd.py --input_file 'path_to_input_file' --reference_file 'path_to_reference_file' --dataset_num '[1-photons, 1-pions, 2, 3]' --mode '[cls-low, clow-low-normed, cls-high]' --binning_file 'xml_binning_files/file_name'
```

**Note**: The samples in a given folder are saved with specific naming convension. Specifically, dataset_n_particle_model.h5, where n stands for the dataset number, partcile stands for type of particle, e.g., electron, and model stands for CaloDiffusion, CaloScore, CaloINN or Geant4. In our evaluation scripts, we assume the saved samples follow this naming convension and based on that we read from the path. For better understanding please refere to [Dataset 2 Samples on Zenodo](https://zenodo.org/records/14883798). 
