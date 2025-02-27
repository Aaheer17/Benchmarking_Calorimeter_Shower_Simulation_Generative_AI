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
