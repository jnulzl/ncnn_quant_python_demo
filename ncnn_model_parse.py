import os
import sys
import numpy as np


def parseConvParam(layer_split):
    # name:[value, id]
    attri_name_value = {
        'num_output': [0, 0], 'kernel_w': [0, 1], 'dilation_w': [1, 2],
        'stride_w': [1, 3], 'pad_left': [0, 4], 'bias_term': [0, 5],
        'weight_data_size': [0, 6], 'int8_scale_term': [0, 8], 'activation_type': [0, 9],
        'activation_params': [[], 10], 'kernel_h': [0, 11], 'dilation_h': [1, 12],
        'stride_h': [1, 13], 'pad_right': [0, 15], 'pad_top': [0, 14],
        'pad_bottom': [0, 16], 'impl_type': [0, 17], 'pad_value': [0.0, 18]
    }

    conv_param_id_to_name = {}
    _param_map = {}

    for (key, val) in attri_name_value.items():
        conv_param_id_to_name[str(val[1])] = key
        _param_map[key] = val[0]

    for item in layer_split:
        if '=' in item:
            item = item.strip().split('=')
            if int(item[0]) < 0:
                continue
            _param_map[conv_param_id_to_name[item[0]]] = int(item[1])

    _param_map['kernel_h'] = _param_map['kernel_w']
    _param_map['dilation_h'] = _param_map['dilation_w']
    _param_map['stride_h'] = _param_map['stride_w']
    _param_map['pad_right'] = _param_map['pad_left']
    _param_map['pad_top'] = _param_map['pad_left']
    _param_map['pad_bottom'] = _param_map['pad_left']

    _param_map['bias_size'] = 0 if 0 == _param_map['bias_term'] else _param_map['num_output']

    return _param_map


def parseDeconvParam(layer_split):
    # name:[value, id]
    attri_name_value = {
        'num_output': [0, 0], 'kernel_w': [0, 1], 'dilation_w': [1, 2],
        'stride_w': [1, 3], 'pad_left': [0, 4], 'bias_term': [0, 5],
        'weight_data_size': [0, 6], 'activation_type': [0, 9],
        'activation_params': [[], 10], 'kernel_h': [0, 11], 'dilation_h': [1, 12],
        'stride_h': [1, 13], 'pad_right': [0, 15], 'pad_top': [0, 14],
        'pad_bottom': [0, 16], 'output_pad_right': [0, 18], 'output_pad_bottom': [0.0, 19],
        'output_w': [0, 20], 'output_h': [0, 21]
    }

    conv_param_id_to_name = {}
    _param_map = {}

    for (key, val) in attri_name_value.items():
        conv_param_id_to_name[str(val[1])] = key
        _param_map[key] = val[0]

    for item in layer_split:
        if '=' in item:
            item = item.strip().split('=')
            if int(item[0]) < 0:
                continue
            _param_map[conv_param_id_to_name[item[0]]] = int(item[1])

    _param_map['kernel_h'] = _param_map['kernel_w']
    _param_map['dilation_h'] = _param_map['dilation_w']
    _param_map['stride_h'] = _param_map['stride_w']
    _param_map['pad_right'] = _param_map['pad_left']
    _param_map['pad_top'] = _param_map['pad_left']
    _param_map['pad_bottom'] = _param_map['pad_left']

    _param_map['output_pad_bottom'] = _param_map['output_pad_right']
    _param_map['output_h'] = _param_map['output_w']

    _param_map['bias_size'] = 0 if 0 == _param_map['bias_term'] else _param_map['num_output']

    return _param_map


def parseDwconvParam(layer_split):
    # name:[value, id]
    attri_name_value = {
        'num_output': [0, 0], 'kernel_w': [0, 1], 'dilation_w': [1, 2],
        'stride_w': [1, 3], 'pad_left': [0, 4], 'bias_term': [0, 5],
        'weight_data_size': [0, 6], 'group': [0, 7], 'int8_scale_term': [0, 8], 'activation_type': [0, 9],
        'activation_params': [[], 10], 'kernel_h': [0, 11], 'dilation_h': [1, 12],
        'stride_h': [1, 13], 'pad_right': [0, 15], 'pad_top': [0, 14],
        'pad_bottom': [0, 16], 'pad_value': [0.0, 18]
    }

    conv_param_id_to_name = {}
    _param_map = {}

    for (key, val) in attri_name_value.items():
        conv_param_id_to_name[str(val[1])] = key
        _param_map[key] = val[0]

    for item in layer_split:
        if '=' in item:
            item = item.strip().split('=')
            _param_map[conv_param_id_to_name[item[0]]] = int(item[1])

    _param_map['kernel_h'] = _param_map['kernel_w']
    _param_map['dilation_h'] = _param_map['dilation_w']
    _param_map['stride_h'] = _param_map['stride_w']
    _param_map['pad_right'] = _param_map['pad_left']
    _param_map['pad_top'] = _param_map['pad_left']
    _param_map['pad_bottom'] = _param_map['pad_left']
    _param_map['bias_size'] = 0 if 0 == _param_map['bias_term'] else _param_map['num_output']

    return _param_map


def parseDeDwconvParam(layer_split):
    # name:[value, id]
    attri_name_value = {
        'num_output': [0, 0], 'kernel_w': [0, 1], 'dilation_w': [1, 2],
        'stride_w': [1, 3], 'pad_left': [0, 4], 'bias_term': [0, 5],
        'weight_data_size': [0, 6], 'group': [1, 7], 'activation_type': [0, 9],
        'activation_params': [[], 10], 'kernel_h': [0, 11], 'dilation_h': [1, 12],
        'stride_h': [1, 13], 'pad_right': [0, 15], 'pad_top': [0, 14],
        'pad_bottom': [0, 16], 'output_pad_right': [0, 18], 'output_pad_bottom': [0.0, 19],
        'output_w': [0, 20], 'output_h': [0, 21]
    }

    conv_param_id_to_name = {}
    _param_map = {}

    for (key, val) in attri_name_value.items():
        conv_param_id_to_name[str(val[1])] = key
        _param_map[key] = val[0]

    for item in layer_split:
        if '=' in item:
            item = item.strip().split('=')
            if int(item[0]) < 0:
                continue
            _param_map[conv_param_id_to_name[item[0]]] = int(item[1])

    _param_map['kernel_h'] = _param_map['kernel_w']
    _param_map['dilation_h'] = _param_map['dilation_w']
    _param_map['stride_h'] = _param_map['stride_w']
    _param_map['pad_right'] = _param_map['pad_left']
    _param_map['pad_top'] = _param_map['pad_left']
    _param_map['pad_bottom'] = _param_map['pad_left']

    _param_map['output_pad_bottom'] = _param_map['output_pad_right']
    _param_map['output_h'] = _param_map['output_w']

    _param_map['bias_size'] = 0 if 0 == _param_map['bias_term'] else _param_map['num_output']

    return _param_map


def parseInnerParam(layer_split):
    attri_name_value = {
        'num_output': [0, 0], 'bias_term': [0, 1], 'weight_data_size': [0, 2],
        'int8_scale_term': [0, 8], 'activation_type': [0, 9], 'activation_params': [[], 10]
    }

    conv_param_id_to_name = {}
    _param_map = {}

    for (key, val) in attri_name_value.items():
        conv_param_id_to_name[str(val[1])] = key
        _param_map[key] = val[0]

    for item in layer_split:
        if '=' in item:
            item = item.strip().split('=')
            _param_map[conv_param_id_to_name[item[0]]] = int(item[1])

    _param_map['bias_size'] = 0 if 0 == _param_map['bias_term'] else _param_map['num_output']

    return _param_map


def parseBatchNormParam(layer_split):
    # name:[value, id]
    attri_name_value = {
        'channels': [0, 0], 'eps': [0.0, 1]
    }

    conv_param_id_to_name = {}
    _param_map = {}

    for (key, val) in attri_name_value.items():
        conv_param_id_to_name[str(val[1])] = key
        _param_map[key] = val[0]

    for item in layer_split:
        if '=' in item:
            item = item.strip().split('=')
            if int(item[0]) < 0:
                continue
            _param_map[conv_param_id_to_name[item[0]]] = int(item[1])

    _param_map['weight_data_size'] = _param_map['channels'] * 4  # slope mean variance bias
    _param_map['bias_size'] = 0

    return _param_map


def parseNCNNParam(ncnn_param_file_path):
    weight_name_attribute = {}
    # See:https://github.com/Tencent/ncnn/wiki/operation-param-weight-table
    valid_weight_layer_type = ['Convolution', 'ConvolutionDepthWise', 'InnerProduct',
                              'BatchNorm', 'Deconvolution', 'DeconvolutionDepthWise']
    invalid_weight_layer_type = ['Dequantize', 'Embed', 'InstanceNorm', 'Normalize',
                                'Padding', 'Requantize', 'Scale', 'PReLU']
    weights_bias_count = 0
    with open(ncnn_param_file_path, 'r') as fpR:
        __tmp = fpR.readlines()
        lines = [item.strip() for item in __tmp]
        layers = lines[2:]
        for layer in layers:
            layer_split = layer.split()
            layer_type = layer_split[0]
            if layer_type in invalid_weight_layer_type:
                return None, None

            if layer_type not in valid_weight_layer_type:
                continue
            layer_name = layer_split[1]
            if 'Convolution' == layer_type:
                layer_attri = parseConvParam(layer_split)
            elif 'ConvolutionDepthWise' == layer_type:
                layer_attri = parseDwconvParam(layer_split)
            elif 'InnerProduct' == layer_type:
                layer_attri = parseInnerParam(layer_split)
            elif 'BatchNorm' == layer_type:
                layer_attri = parseBatchNormParam(layer_split)
            elif 'Deconvolution' == layer_type:
                layer_attri = parseDeconvParam(layer_split)
            elif 'DeconvolutionDepthWise' == layer_type:
                layer_attri = parseDeDwconvParam(layer_split)

            weights_bias_count += layer_attri['weight_data_size']
            weights_bias_count += layer_attri['bias_size']
            '''
            [flag] (optional)
            [raw data]
            [padding] (optional)
            '''
            if 'BatchNorm' != layer_type:
                weights_bias_count += 1  # flag preifx

            layer_attri['layer_type'] = layer_type
            weight_name_attribute[layer_name] = layer_attri

    return weight_name_attribute, weights_bias_count


def parseNCNNModel(ncnn_param_file_path, ncnn_fp32_bin_file_path):
    weight_name_attribute, weights_bias_count = parseNCNNParam(ncnn_param_file_path)
    if weights_bias_count is None:
        return None

    weights = np.fromfile(ncnn_fp32_bin_file_path, dtype=np.float32)
    begin_index = 0
    end_index = 0
    for key, item in weight_name_attribute.items():
        layer_type = item['layer_type']
        if 'BatchNorm' == layer_type:
            begin_index = end_index
            end_index = begin_index + item['weight_data_size']
            weight_name_attribute[key]['weight_data'] = None
            weight_name_attribute[key]['bias_data'] = None
        else:
            begin_index = (end_index + 1)
            end_index = begin_index + item['weight_data_size']
            layer_weight = weights[begin_index:end_index]
            begin_index = end_index
            end_index = begin_index + item['bias_size']
            layer_bias = weights[begin_index:end_index]
            weight_name_attribute[key]['weight_data'] = layer_weight
            weight_name_attribute[key]['bias_data'] = layer_bias
            # print(weight_name_attribute[key]['layer_name'] + " weight : ", layer_weight.shape, layer_weight)
            # print(weight_name_attribute[key]['layer_name'] + " bias : ", layer_bias.shape, layer_bias)

    return weight_name_attribute


if __name__ == '__main__':
    ncnn_param_file_path = "best_320_opt.param"
    ncnn_fp32_bin_file_path = "best_320_opt.bin"
    weight_name_attribute = parseNCNNModel(ncnn_param_file_path, ncnn_fp32_bin_file_path)
    for key, item in weight_name_attribute.items():
        print(item)
        # print(item['layer_weight'])
        # print(item['layer_bias'])
