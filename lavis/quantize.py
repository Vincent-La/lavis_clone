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
                        ['attn', 'mlp'],
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
    
    # cls BERT output
    if args['qformer_cls_modules']:
        for module in args['qformer_cls_modules']:
            if module == 'transform':
                weight_bits = args['qformer_cls_transform_weight_bits']
                quantize_selected_modules(model.Qformer.cls.predictions,
                                          ['dense'],
                                          weight_bits)
            elif module == 'decoder':
                weight_bits = args['qformer_cls_decoder_weight_bits']
                quantize_selected_modules(model.Qformer.cls.predictions,
                                          module,
                                          weight_bits)
            
    # final output layers
    if args['output_modules']:
        for module in args['output_modules']:
            if module == 'vision_proj':
                weight_bits = args['vision_proj_weight_bits']
            elif module == 'text_proj':
                weight_bits = args['text_proj_weight_bits']
            elif module == 'itm_head':
                weight_bits = args['itm_head_weight_bits']
            
            quantize_selected_modules(model,
                                      [module],
                                      weight_bits)
    
       
def model_size(model):
    # returns all layers of model
    def get_layers(model):
        children = list(model.children())
        return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]
    
    layers = get_layers(model)
    size = 0
    
    # model params
    for layer in layers:
        for name, param in layer.named_parameters():
            #  NOTE: element_size in bits
            element_size = layer.weight_bits if isinstance(layer, NBitLinearDynamic) else (param.element_size() * 8)
            size += param.nelement() * element_size
    
    # model buffers (not quantized)
    for buffer in model.buffers():
        size += buffer.nelement() * (buffer.element_size() * 8)

    # bits --> megabytes
    size /= (8e6)
    return size
