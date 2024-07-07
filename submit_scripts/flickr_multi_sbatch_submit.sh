python flickr_multi_sbatch.py --env vit_small_batch \
                              --nhrs 8 \
                              --qos scav \
                              --partition nexus \
                              --gpu 8 --gpu-type a5000 a6000 \
                              --cores 1 \
                              --mem 128 \
                              --base-dir /fs/nexus-scratch/vla/LAVIS \
                              --output-dirname slurm_files \
                              --dryrun

# TODO: set base-dir to project-dir