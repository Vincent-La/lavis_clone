#!/bin/bash

#SBATCH --job-name=blip2_flickr_argstest                    # sets the job name
#SBATCH --output=blip2_flickr_argstest.%j                    # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=blip2_flickr_argstest.%j                     # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=20:00:00                                      # how long you would like your job to run; format=hh:mm:ss

#SBATCH --array=1-1%80
#SBATCH --requeue
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtxa5000:8


source ~/.bashrc
# eval "$(micromamba shell hook --shell bash)"
micromamba activate LAVIS

python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path ret_flickr_eval.yaml --visual-encoder-block-modules qkv,proj,fc1,fc2 --visual-encoder-block-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38 --visual-encoder-block-weight-bits 8


wait                                                            # wait for any background processes to complete

# once the end of the batch script is reached your job allocation will be revoked
