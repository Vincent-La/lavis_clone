import torch
import torch.nn as nn
from lavis.layers.nbitlineardynamic import NBitLinearDynamic

'''
Takes in nn.Linear and returns equivalent NBitLinearDynamic replacement
'''
def quantize_layer(module:nn.Linear, weight_bits = 32, activation_bits=32):
    
    with torch.no_grad():
        
        bias = True if module.bias != None else False
        
        Q_layer = NBitLinearDynamic(module.in_features, 
                                    module.out_features, 
                                    bias=bias,
                                    weight_bits = weight_bits,
                                    activation_bits = activation_bits)

        # copy over weights
        Q_layer.weight.copy_(module.weight)
        if bias:
            Q_layer.bias.copy_(module.bias)

    return Q_layer


# def quantize_visual_encoder_block(module_parent, args):
#     for name, module in module_parent.named_children():
#         if name in args['visual_encoder_block_modules']:
#             print('parent: ', module_parent)
#             print('child: ', name)
            
#             # TODO: could customize weight_bits/activation_bits per block 
#             setattr(module_parent, name, quantize_layer(module, weight_bits = args['visual_encoder_block_weight_bits']))
            
#         else:
#             quantize_visual_encoder_block(module, args)
            

# def quantize_visual_encoder_blocks(blocks, args):
#     for name, module in blocks.named_children():
#         # apply quant config to specified block indices
#         if int(name) in args['visual_encoder_block_indices']:
#             # print('here')
#             quantize_visual_encoder_block(module, args)
         


def quantize_selected_modules(module_cur, modules_to_quant, weight_bits):
    for name, module in module_cur.named_children():
        if name in modules_to_quant:
            setattr(module_cur, name, quantize_layer(module, weight_bits = weight_bits))
        else:
            quantize_selected_modules(module, modules_to_quant, weight_bits)

def quantize_block(block, parent_modules, modules_to_quant, weight_bits):
    for name, module in block.named_children():
        # only continue recursively if specified block-level module 
        if name in parent_modules:
            quantize_selected_modules(module, modules_to_quant, weight_bits)
            

# quantize specified block/layer indices
# TODO: could customize weight_bits/activation_bits per block 
def quantize_blocks(blocks, parent_modules, modules_to_quant, indices, weight_bits):
    for name, module in blocks.named_children():
         # apply quant config to specified block indices
        if int(name) in indices:
            quantize_block(module, parent_modules, modules_to_quant, weight_bits)
        
def quantize(model, args):
    
    args = vars(args)
    
    # Visual encoder blocks
    if args['visual_encoder_block_modules']:
        quantize_blocks(model.visual_encoder.blocks, 
                        ['attn'],
                        args['visual_encoder_block_modules'],
                        args['visual_encoder_block_indices'],
                        args['visual_encoder_block_weight_bits'])
    
    # Q-former blocks
    if args['qformer_layer_indices']:
        
        qformer_layers = model.Qformer.bert.encoder.layer
        
        # Self-attention
        if args['qformer_self_attention_modules']:
            quantize_blocks(qformer_layers,
                            ['attention'],
                            args['qformer_self_attention_modules'],
                            args['qformer_layer_indices'],
                            args['qformer_self_attention_weight_bits'])
        
        # Cross-attention
        if args['qformer_cross_attention_modules']:
            quantize_blocks(qformer_layers,
                            ['crossattention'],
                            args['qformer_cross_attention_modules'],
                            args['qformer_layer_indices'],
                            args['qformer_cross_attention_weight_bits'])
        
       
       # text sub-module feed forward
        if args['qformer_img_ff_modules']:
            quantize_blocks(qformer_layers,
                            args['qformer_text_ff_modules'],
                            ['dense'],
                            args['qformer_layer_indices'],
                            args['qformer_text_ff_weight_bits'])
       
        # img sub-module feed forward
        if args['qformer_img_ff_modules']:
            quantize_blocks(qformer_layers,
                            args['qformer_img_ff_modules'],
                            ['dense'],
                            args['qformer_layer_indices'],
                            args['qformer_img_ff_weight_bits'])
        
        
    # # Visual encoder blocks
    # if args['visual_encoder_block_modules']:
    #     quantize_visual_encoder_blocks(model.visual_encoder.blocks, args)
        
    # # TODO: Q-Former layers
    
        
    
        
