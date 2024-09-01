#!/bin/bash

#SBATCH -A m4392                 
#SBATCH -C gpu&hbm80g
#SBATCH -q shared                  
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32          
#SBATCH --time=00:30:00                   
#SBATCH --output=/global/homes/s/samyak09/GSOC-SR/GSOC-24-SR/GP+Beam_search/runs/runs%j.out            
#SBATCH --error=/global/homes/s/samyak09/GSOC-SR/GSOC-24-SR/GP+Beam_search/runs/runs%j.err             
#SBATCH --mail-type=END,FAIL              
#SBATCH --mail-user=samyakjha71@gmail.com 

module load python pytorch/2.0            # Load necessary modules

# Install the required Python packages
pip install deap

# Run your Python script
srun python main.py
