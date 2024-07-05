import argparse
from itertools import chain

def list_str(values):
    return values.split(',')

def list_int(values):
    to_ret = list(map(int, values.split(',')))
    if(len(to_ret) == 1):
        return to_ret[0]
    
    return to_ret

def args_parser():
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
                        # nargs="*",
                        # choices= ['qkv', 'proj', 'fc1', 'fc2'],
                        default=None,
                        type=list_str,                        
                        help='modules of visual-encoder blocks to quantize')
    
    parser.add_argument('--visual-encoder-block-indices',
                         required=False,
                        #  nargs='*',
                         type=list_int,
                        #  choices= [i for i in range(39)],   # NOTE: can enforce hard-coded number of possible blocks for ViT
                         default=None,      
                         help = 'indices of visual-encoder blocks to quantize')
    
    parser.add_argument('--visual-encoder-block-weight-bits',
                        required=False,
                        type=int,
                        choices=[i for i in range(1,9)],
                        default=None,
                        help = 'weight bits for visual-encoder blocks')
    
    # Q-former options
    parser.add_argument('--qformer-layer-indices',
                        required= False,
                        default = None,
                        # nargs='*',
                        type = list_int,
                        # choices = [i for range(12)],   # NOTE: can enforce hard-coded number of possible blocks in Q-former
                        help = 'indices of Q-former to quantize')   
    
    # self-attention options
    parser.add_argument('--qformer-self-attention-modules',
                        required=False,
                        default = None,
                        # nargs='*',
                        # choices=['query', 'key', 'value', 'dense'],
                        type=list_str,
                        help = 'self-attention Q-Former modules to quantize (shared by img + text submodules) (per-block)')   # NOTE: 'dense' refers to output linear layer for BertLayer
    
    parser.add_argument('--qformer-self-attention-weight-bits',
                        required=False,
                        type=int,
                        default=None,
                        choices=[i for i in range(1,9)],
                        help='weight bits for Q-Former self attention modules')
    
    # cross attention options
    parser.add_argument('--qformer-cross-attention-modules',
                        required=False,
                        default = None,
                        type=list_str,
                        # nargs = '*',
                        # choices=['query', 'key', 'value', 'dense'], # NOTE: 'dense' refers to output linear layer for BertLayer
                        help = 'cross-attention Q-former modules to quantize (img submodule) (per-block)')

    parser.add_argument('--qformer-cross-attention-weight-bits',
                        required=False,
                        type=list_int,
                        default=None,
                        # choices=[i for i in range(1,9)],
                        help='weight bits for Q-Former cross attention modules')

    # feed-forward options
    
    parser.add_argument('--qformer-text-ff-modules',
                        required=False,
                        default=None,
                        type=list_str,
                        # nargs = '*',
                        # choices=['intermediate', 'output'],
                        help='modules of Q-Former text submodule feed forward to quantize (per-block)')
    
    parser.add_argument('--qformer-text-ff-weight-bits',
                        required=False,
                        type=list_int,
                        default=None,
                        # choices=[i for i in range(1,9)],
                        help='weight bits for Q-Former text submodule feed-forward (intermediate + output layers)')
    
    parser.add_argument('--qformer-img-ff-modules',
                    required=False,
                    default=None,
                    type=list_str,
                    # nargs = '*',
                    # choices=['intermediate_query', 'output_query'],
                    help='modules of Q-Former image submodule feed forward to quantize (per-block)')
     
    parser.add_argument('--qformer-img-ff-weight-bits',
                        required=False,
                        type=list_int,
                        default=None,
                        # choices=[i for i in range(1,9)],
                        help='weight bits for Q-Former img submodule feed-forward (intermediate + output layers)')
    
    
    # options for cls modules
    parser.add_argument('--qformer-cls-modules',
                        required=False,
                        default=None,
                        type=list_str,
                        # nargs='*',
                        # choices = ['transform', 'decoder'],
                        help = 'modules of Q-Former [CLS] (BertOnlyMLMHead) head to quantize')
    
    parser.add_argument('--qformer-cls-transform-weight-bits',
                        required=False,
                        default=None,
                        type=list_int,
                        # choices=[i for i in range(1,9)],
                        help = 'weight bits for Q-Former [CLS] (BertOnlyMLMHead) transform layer')
    
    parser.add_argument('--qformer-cls-decoder-weight-bits',
                    required=False,
                    default=None,
                    type=list_int,
                    # choices=[i for i in range(1,9)],
                    help = 'weight bits for Q-Former [CLS] (BertOnlyMLMHead) decoder layer')
    
    # options for final output/projection layers
    parser.add_argument('--output-modules',
                        required=False,
                        default = None,
                        type = list_str,
                        # nargs = '*',
                        # choices=['vision_proj', 'text_proj', 'itm_head'],          # NOTE: itm_head is for image-text matching
                        help='output modules to quantize')
    
    parser.add_argument('--vision-proj-weight-bits',
                        required=False,
                        default=None,
                        type = list_int,
                        # choices=[i for i in range(1,9)],
                        help='weight bits for final vision_proj layer')
    
    parser.add_argument('--text-proj-weight-bits',
                        required=False,
                        default=None,
                        type = list_int,
                        # choices=[i for i in range(1,9)],
                        help='weight bits for final text_proj layer')
    
    parser.add_argument('--itm-head-weight-bits',
                        required=False,
                        default=None,
                        type = list_int,
                        # choices=[i for i in range(1,9)],
                        help='weight bits for final itm_head layer')
    
    
    return parser

def validate_args(args):
    return
    args_dict = vars(args)
    
     # TODO: ensure groups of settings are all defined, either all None or all defined
    # if None in [args_dict['visual_encoder_block_modules'],
    #             args_dict['visual_encoder_block_indices'], 
    #             args_dict['visual_encoder_block_weight_bits']]:
        
    #     parser.error('--visual-encoder-block-modules, --visual-encoder-block-indices, --visual-encoder-block-weight-bits, must be given together')
    

def parse_args():
    
    # split this up so that its easier to test in a notebook
    parser = args_parser()
    
    args = parser.parse_args()
    validate_args(args)
    
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    

    return args