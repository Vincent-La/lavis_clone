python nbit_vit_flickr_multi_sbatch.py --env nbit_vit \
                                       --nhrs 8 \
                                       --qos scav \
                                       --partition nexus \
                                       --gpu 8 --gpu-type a5000 a6000 \
                                       --cores 1 \
                                       --mem 128 \
                                       --output-dirname slurm_files \
                                       --filename submit.sh \

# NOTE: set base-dir to repo top-level dir in nbit_flickr_multi_sbatch.py
