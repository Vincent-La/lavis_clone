python multi_sbatch.py --nhrs 8 \
                       --output-dirname slurm_files \
                       --qos scav \
                       --partition nexus \
                       --env exp_name \
                       --gpu 8 --gpu-type a4000 \
                       --cores 1 \
                       --mem 128 \
                       --dryrun
                       