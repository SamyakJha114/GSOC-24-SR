#!/bin/bash

#SBATCH -A m4392                 
#SBATCH -C gpu&hbm80g
#SBATCH -q shared                  
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64          
#SBATCH --time=09:00:00                   
#SBATCH --output=/global/homes/s/samyak09/GSOC-SR/GSOC-24-SR/GP+Beam_search/runs/runs_total%j.out            
#SBATCH --error=/global/homes/s/samyak09/GSOC-SR/GSOC-24-SR/GP+Beam_search/runs/runs_total%j.err             
#SBATCH --mail-type=END,FAIL              
#SBATCH --mail-user=samyakjha71@gmail.com 

module load python/3.11
# module load python pytorch/2.0            # Load necessary modules

# Install the required Python packages
pip install torch 
pip install deap

# Run your Python script
srun python GSOC-24-SR/GP+Beam_search/main.py
