import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    # TODO: maybe add more granularity for visual encoder options (choice of bitwidth for each block module)
    # vision encoder options
    parser.add_argument('--visual-encoder-block-modules', 
                        required=False,
                        nargs="*",
                        choices= ['qkv', 'proj', 'fc1', 'fc2'],
                        default=None,                         
                        help='modules of visual-encoder blocks to quantize')
    
    parser.add_argument('--visual-encoder-block-indices',
                         required=False,
                         nargs='*',
                         type=int,
                        #  choices= [i for i in range(39)],   # NOTE: can enforce hard-coded number of possible blocks for ViT
                         default=None,      
                         help = 'indices of visual-encoder blocks to quantize')
    
    parser.add_argument('--visual-encoder-block-weight-bits',
                        required=False,
                        type=int,
                        choices=[1,2,4,6,8],
                        default=None,
                        help = 'weight bits for visual-encoder blocks')
    
    # Q-former options
    parser.add_argument('--qformer-layer-indices',
                        required= False,
                        default = None,
                        nargs='*',
                        type = int,
                        # choices = [i for range(12)],   # NOTE: can enforce hard-coded number of possible blocks in Q-former
                        help = 'indices of Q-former to quantize')   
    
    # self-attention options
    parser.add_argument('--qformer-self-attention-modules',
                        required=False,
                        default = None,
                        nargs='*',
                        choices=['query', 'key', 'value', 'dense'],
                        help = 'self-attention Q-Former modules to quantize (shared by img + text submodules) (per-block)')   # NOTE: 'dense' refers to output linear layer for BertLayer
    
    parser.add_argument('--qformer-self-attention-weight-bits',
                        required=False,
                        type=int,
                        default=None,
                        choices=[1,2,4,6,8],
                        help='weight bits for Q-Former self attention modules')
    
    # cross attention options
    parser.add_argument('--qformer-cross-attention-modules',
                        required=False,
                        default = None,
                        nargs = '*',
                        choices=['query', 'key', 'value', 'dense'], # NOTE: 'dense' refers to output linear layer for BertLayer
                        help = 'cross-attention Q-former modules to quantize (img submodule) (per-block)')

    parser.add_argument('--qformer-cross-attention-weight-bits',
                        required=False,
                        type=int,
                        default=None,
                        choices=[1,2,4,6,8],
                        help='weight bits for Q-Former cross attention modules')

    # feed-forward options
    parser.add_argument('--qformer-text-ff-intermediate',
                        required=False,
                        default = None,
                        choices=['dense'],
                        help= 'intermediate module of Q-former text submodule feed-forward to quantize (per-block)')

    parser.add_argument('--qformer-text-ff-output',
                        required=False,
                        default=None,
                        choices=['dense'],
                        help='output module of Q-former text submodule feed-forward to quantize (per-block)')
    
    parser.add_argument('--qformer-text-ff-weight-bits',
                        required=False,
                        type=int,
                        default=None,
                        choices=[1,2,4,6,8],
                        help='weight bits for Q-Former text submodule feed-forward (intermediate + output layers)')
    
    parser.add_argument('--qformer-img-ff-intermediate',
                        required=False,
                        default = None,
                        choices=['dense'],
                        help= 'intermediate module of Q-former img submodule feed-forward to quantize (per-block)')
    
    parser.add_argument('--qformer-img-ff-output',
                        required=False,
                        default = None,
                        choices=['dense'],
                        help= 'output module of Q-former img submodule feed-forward to quantize (per-block)')
    
    parser.add_argument('--qformer-img-ff-weight-bits',
                        required=False,
                        type=int,
                        default=None,
                        choices=[1,2,4,6,8],
                        help='weight bits for Q-Former img submodule feed-forward (intermediate + output layers)')
    
    
    # options for final output/projection layers
    parser.add_argument('--qformer-output-modules',
                        required=False,
                        default = None,
                        choices=['vision_proj', 'text_proj', 'itm_head'],          # NOTE: itm_head is for image-text matching
                        help='output modules of Q-former to quantize')
    
    parser.add_argument('--qformer-vision-proj-weight-bits',
                        required=False,
                        default=None,
                        choices=[1,2,4,6,8],
                        help='weight bits for Q-Former final vision_proj layer')
    
    parser.add_argument('--qformer-text-proj-weight-bits',
                        required=False,
                        default=None,
                        choices=[1,2,4,6,8],
                        help='weight bits for Q-Former final text_proj layer')
    
    parser.add_argument('--qformer-itm-head-weight-bits',
                        required=False,
                        default=None,
                        choices=[1,2,4,6,8],
                        help='weight bits for Q-Former final itm_head layer')
    
    
    # TODO: options for cls modules (only for LLM ?)

    # NOTE:uncomment this when done 
    args = parser.parse_args()

    # TODO: REMOVE
    # CLI_INPUT = f'''
    #             --cfg-path /nfshomes/vla/scratch/LAVIS/ret_flickr_eval.yaml
    #             --visual-encoder-block-modules qkv proj fc1 fc2
    #             --visual-encoder-block-indices {' '.join([str(i) for i in range(39)])}
    #             --visual-encoder-block-weight-bits 8
    #             --qformer-self-attention-modules query key value dense
    #             --qformer-self-attention-weight-bits 8
    #             --qformer-cross-attention-modules query key value dense
    #             --qformer-cross-attention-weight-bits 8
    #             '''
    
    # args = parser.parse_args(CLI_INPUT.split())
   
    
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    args_dict = vars(args)
    
    # TODO: ensure groups of settings are all defined
    # if None in [args_dict['visual_encoder_block_modules'],
    #             args_dict['visual_encoder_block_indices'], 
    #             args_dict['visual_encoder_block_weight_bits']]:
        
    #     parser.error('--visual-encoder-block-modules, --visual-encoder-block-indices, --visual-encoder-block-weight-bits, must be given together')
    

    return args