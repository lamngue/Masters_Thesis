#!/bin/bash
#SBATCH --account=cseduimc030
#SBATCH --partition=csedu-prio,csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

# Commands to run your program go here, e.g.:
python Thesis_Train_Model.py