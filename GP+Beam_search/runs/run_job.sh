#!/bin/bash
#SBATCH --job-name=gp_run                 # Job name
#SBATCH --account=m4392                   # Project account
#SBATCH --partition=cpu                   # Use CPU partition
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=64              # Number of CPU cores per node
#SBATCH --time=00:30:00                   # Time limit (hh:mm:ss)
#SBATCH --output=gp_run_%j.out            # Standard output and error log
#SBATCH --error=gp_run_%j.err             # Error log
#SBATCH --mail-type=END,FAIL              # Notifications for job done & fail
#SBATCH --mail-user=your_email@domain.com # Your email for notifications

module load python pytorch/2.0            # Load necessary modules

# Install the required Python packages
pip install deap

# Run your Python script
srun python main.py --file_index=0        # Adjust the file_index as needed
