import torch
import torch.nn as nn
from layers.nbitlineardynamic import NBitLinearDynamic

'''
Takes in nn.Linear and returns equivalent NBitLinearDynamic replacement
'''
def quantize_layer(module:nn.Linear, weight_bits = 32, activation_bits=32):
    
    print('weight_bits: ', weight_bits)
    
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


def quantize_visual_encoder_block(module_parent, args):
    for name, module in module_parent.named_children():
        if name in args['visual_encoder_block_modules']:
            print('parent: ', module_parent)
            print('child: ', name)
            
            # TODO: could customize weight_bits/activation_bits per block 
            setattr(module_parent, name, quantize_layer(module, weight_bits = args['visual_encoder_block_weight_bits']))
            
        else:
            quantize_visual_encoder_block(module, args)
            

def quantize_visual_encoder_blocks(blocks, args):
    for name, module in blocks.named_children():
        # apply quant config to specified block indices
        if int(name) in args['visual_encoder_block_indices']:
            # print('here')
            quantize_visual_encoder_block(module, args)
         


def quantize(model, args):
    # Visual encoder blocks
    if args['visual_encoder_block_modules']:
        quantize_visual_encoder_blocks(model.visual_encoder.blocks, args)
        
    # TODO: Q-Former layers
    
        
    
        
