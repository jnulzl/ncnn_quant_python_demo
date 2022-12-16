'''
ncnn model quant(ncnn2table.cpp --> ncnn2table.py)

retf: https://zhuanlan.zhihu.com/p/387072703
retf: https://github.com/Tencent/ncnn/blob/eaab441d3c/tools/quantize/ncnn2table.cpp
retf: https://github.com/lutzroeder/netron/blob/main/source/ncnn.js
retf: https://github.com/Tencent/ncnn/wiki/operation-param-weight-table
retf: https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
retf: https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
'''
import os
import sys
import cv2
import numpy as np
import ncnn
from ncnn_model_parse import parseNCNNModel


class QuantBlobStat(object):

    def __init__(self):
        super(QuantBlobStat, self).__init__()
        self.threshold = 0.0
        self.absmax = 0.0
        # ACIQ
        self.total = 0
        self.histogram = []
        self.histogram_normed = []


def compute_kl_divergence(veca, vecb):
    sizeab = len(veca)
    res = 0
    for idx in range(sizeab):
        res += veca[idx] * np.log(veca[idx] / vecb[idx])
    return res


class QuantNet(ncnn.Net):

    def __init__(self, input_param, input_bin,
                 quant_sample_list_txts,
                 means=[[0.0, 0.0, 0.0]],
                 norms=[[1.0, 1.0, 1.0]],
                 shapes=[[320, 320, 3]],
                 type_to_pixels=[1],
                 quantize_num_threads=4):
        super(QuantNet, self).__init__()
        self.input_param = input_param
        self.input_bin = input_bin
        self.weight_name_attribute = parseNCNNModel(self.input_param, self.input_bin)
        self.listspaths = []
        for quant_sample_list_txt in quant_sample_list_txts:
            with open(quant_sample_list_txt, 'r') as fpR:
                __tmp = fpR.readlines()
                _tmp = [item.strip() for item in __tmp]
                self.listspaths.append(_tmp)
        self.means = means
        self.norms = norms
        self.shapes = shapes
        self.type_to_pixels = type_to_pixels
        self.quantize_num_threads = ncnn.get_cpu_count() if quantize_num_threads <= 0 else quantize_num_threads

    def init(self):
        self.load_param(self.input_param)
        self.load_model(self.input_bin)
        self.input_blobs = []
        self.conv_layers = []
        self.conv_bottom_blobs = []
        self.conv_top_blobs = []

        # find all input&conv layers
        for index, layer in enumerate(self.layers()):
            if "Input" == layer.type:
                self.input_blobs.append(layer.tops[0])
            elif "Convolution" == layer.type or "ConvolutionDepthWise" == layer.type or "InnerProduct" == layer.type:
                self.conv_layers.append(index)
                self.conv_bottom_blobs.append(layer.bottoms[0])
                self.conv_top_blobs.append(layer.tops[0])

        # result
        # self.quant_blob_stats = [QuantBlobStat()] * len(self.conv_bottom_blobs) # invalid
        self.quant_blob_stats = [QuantBlobStat() for _ in self.conv_bottom_blobs]
        self.weight_scales = [None] * len(self.conv_layers)
        self.bottom_blob_scales = [None] * len(self.conv_bottom_blobs)

    def quantize_KL(self, num_histogram_bins=2048):
        input_blob_count = len(self.input_blobs)
        conv_layer_count = len(self.conv_layers)
        conv_bottom_blob_count = len(self.conv_bottom_blobs)
        image_count = len(self.listspaths[0])

        # Initialize conv weight scales
        for index in range(conv_layer_count):
            layer = self.layers()[self.conv_layers[index]]
            if "Convolution" == layer.type:
                convolution = self.weight_name_attribute[layer.name]

                num_output = convolution['num_output']
                kernel_w = convolution['kernel_w']
                kernel_h = convolution['kernel_h']
                dilation_w = convolution['dilation_w']
                dilation_h = convolution['dilation_h']
                stride_w = convolution['stride_w']
                stride_h = convolution['stride_h']

                weight_data_size_output = int(convolution['weight_data_size'] / num_output)

                # int8 winograd F43 needs weight data to use 6bit quantization
                quant_6bit = False
                if kernel_w == 3 and kernel_h == 3 and dilation_w == 1 and \
                        dilation_h == 1 and stride_w == 1 and stride_h == 1:
                    quant_6bit = True
                self.weight_scales[index] = [None] * num_output
                weight_data = convolution['weight_data']
                for n in range(num_output):
                    begin_index = weight_data_size_output * n
                    end_index = weight_data_size_output * (n + 1)
                    if quant_6bit:
                        self.weight_scales[index][n] = 31 / np.max(np.abs(weight_data[begin_index: end_index]))
                    else:
                        self.weight_scales[index][n] = 127 / np.max(np.abs(weight_data[begin_index: end_index]))

            elif "ConvolutionDepthWise" == layer.type:
                convolutiondepthwise = self.weight_name_attribute[layer.name]
                group = convolutiondepthwise['group']
                weight_data_size_output = int(convolutiondepthwise['weight_data_size'] / group)

                self.weight_scales[index] = [None] * group
                weight_data = convolutiondepthwise['weight_data']
                for n in range(group):
                    begin_index = weight_data_size_output * n
                    end_index = weight_data_size_output * (n + 1)
                    self.weight_scales[index][n] = 127 / np.max(np.abs(weight_data[begin_index: end_index]))

            elif "InnerProduct" == layer.type:
                innerproduct = self.weight_name_attribute[layer.name]
                num_output = innerproduct['num_output']
                weight_data_size_output = int(innerproduct['weight_data_size'] / num_output)

                self.weight_scales[index] = [None] * num_output
                weight_data = innerproduct['weight_data']
                for n in range(num_output):
                    begin_index = weight_data_size_output * n
                    end_index = weight_data_size_output * (n + 1)
                    self.weight_scales[index][n] = 127 / np.max(np.abs(weight_data[begin_index: end_index]))

        # count the absmax
        for idx in range(image_count):
            ex = self.create_extractor()

            # set input data
            for idy in range(input_blob_count):
                img_path = self.listspaths[idy][idx]
                shape = self.shapes[idy]
                type_to_pixel = self.type_to_pixels[idy]
                mean_vals = self.means[idy]
                norm_vals = self.norms[idy]

                target_w = shape[0]
                target_h = shape[1]
                img = cv2.imread(img_path, 1)
                mat_in = ncnn.Mat.from_pixels_resize(
                    img,
                    type_to_pixel,  # ncnn.Mat.PixelType.PIXEL_BGR,
                    img.shape[1],
                    img.shape[0],
                    target_w,
                    target_h,
                )
                mat_in.substract_mean_normalize(mean_vals, norm_vals)
                ex.input(self.input_blobs[idy], mat_in)

            for idy in range(conv_bottom_blob_count):
                ret, mat_out = ex.extract(self.conv_bottom_blobs[idy])
                max_abs_out = np.max(np.abs(np.array(mat_out)))
                self.quant_blob_stats[idy].absmax = max(self.quant_blob_stats[idy].absmax, max_abs_out)

        # initialize histogram
        for idx in range(conv_bottom_blob_count):
            self.quant_blob_stats[idx].histogram = np.zeros(num_histogram_bins, dtype=np.float32)
            self.quant_blob_stats[idx].histogram_normed = np.zeros(num_histogram_bins, dtype=np.float32)

        # build histogram(no norm)
        for idx in range(image_count):
            ex = self.create_extractor()
            # set input data
            for idy in range(input_blob_count):
                img_path = self.listspaths[idy][idx]
                shape = self.shapes[idy]
                type_to_pixel = self.type_to_pixels[idy]
                mean_vals = self.means[idy]
                norm_vals = self.norms[idy]

                target_w = shape[0]
                target_h = shape[1]
                img = cv2.imread(img_path, 1)
                mat_in = ncnn.Mat.from_pixels_resize(
                    img,
                    type_to_pixel,  # ncnn.Mat.PixelType.PIXEL_BGR,
                    img.shape[1],
                    img.shape[0],
                    target_w,
                    target_h,
                )

                mat_in.substract_mean_normalize(mean_vals, norm_vals)

                ex.input(self.input_blobs[idy], mat_in)

            for idy in range(conv_bottom_blob_count):
                ret, mat_out = ex.extract(self.conv_bottom_blobs[idy])
                out_flatten_abs_array = np.abs(np.array(mat_out).flatten())
                # count histogram bin
                absmax = self.quant_blob_stats[idy].absmax

                # cal hist using numpy.histogram
                out_flatten_abs_equal_zero_count = np.sum(out_flatten_abs_array < 1e-9)
                out_flatten_abs_array = out_flatten_abs_array * num_histogram_bins / absmax
                mask = out_flatten_abs_array >= (num_histogram_bins - 1)
                out_flatten_abs_array[mask] = (num_histogram_bins - 1)
                out_flatten_abs_array = out_flatten_abs_array.astype(np.int32)
                hist, bin_edges = np.histogram(out_flatten_abs_array, bins=np.arange(num_histogram_bins + 1))
                hist[0] -= out_flatten_abs_equal_zero_count

                self.quant_blob_stats[idy].histogram += hist

        # using kld to find the best threshold value
        for idx in range(conv_bottom_blob_count):
            # normalize histogram bin
            self.quant_blob_stats[idx].histogram_normed = self.quant_blob_stats[idx].histogram / np.sum(self.quant_blob_stats[idx].histogram)
            target_bin = 128
            target_threshold = target_bin
            min_kl_divergence = 1e9
            for threshold in range(target_bin, num_histogram_bins):
                kl_eps = 1e-4
                clip_distribution = np.ones(threshold) * kl_eps
                clip_distribution += self.quant_blob_stats[idx].histogram_normed[:threshold]
                clip_distribution[threshold - 1] += np.sum(self.quant_blob_stats[idx].histogram_normed[threshold:])
                # ---------------------------------quantize_distribution---------------------------------
                # process lower edge
                num_per_bin = threshold / target_bin
                quantize_distribution = np.zeros(target_bin, dtype=np.float32)
                end = num_per_bin
                right_lower = int(np.floor(end))
                right_scale = end - right_lower
                if right_scale > 0:
                    quantize_distribution[0] += right_scale * self.quant_blob_stats[idx].histogram_normed[right_lower]
                quantize_distribution[0] += np.sum(self.quant_blob_stats[idx].histogram_normed[:right_lower])
                quantize_distribution[0] /= num_per_bin

                # process normal value
                for j in range(1, target_bin - 1):
                    start = j * num_per_bin
                    end = (j + 1) * num_per_bin

                    left_upper = int(np.ceil(start))
                    left_scale = left_upper - start

                    right_lower = int(np.floor(end))
                    right_scale = end - right_lower
                    if left_scale > 0:
                        quantize_distribution[j] += left_scale * self.quant_blob_stats[idx].histogram_normed[left_upper-1]
                    if right_scale > 0:
                        quantize_distribution[j] += right_scale * self.quant_blob_stats[idx].histogram_normed[right_lower]
                    quantize_distribution[j] += np.sum(self.quant_blob_stats[idx].histogram_normed[left_upper:right_lower])
                    quantize_distribution[j] /= num_per_bin

                # process upper edge
                start = threshold - num_per_bin
                left_upper = int(np.ceil(start))
                left_scale = left_upper - start
                if left_scale > 0:
                    quantize_distribution[target_bin - 1] += left_scale * self.quant_blob_stats[idx].histogram_normed[left_upper-1]
                quantize_distribution[target_bin - 1] += np.sum(self.quant_blob_stats[idx].histogram_normed[left_upper:threshold])
                quantize_distribution[target_bin - 1] /= num_per_bin

                # ---------------------------------expand_distribution---------------------------------
                # process lower edge
                expand_distribution = kl_eps * np.ones(threshold, dtype=np.float32)
                end = num_per_bin
                right_lower = int(np.floor(end))
                right_scale = end - right_lower
                if right_scale > 0:
                    expand_distribution[right_lower] += right_scale * quantize_distribution[0]
                expand_distribution[:right_lower] += quantize_distribution[0]

                # process normal value
                for j in range(1, target_bin - 1):
                    start = j * num_per_bin
                    end = (j + 1) * num_per_bin

                    left_upper = int(np.ceil(start))
                    left_scale = left_upper - start

                    right_lower = int(np.floor(end))
                    right_scale = end - right_lower
                    if left_scale > 0:
                        expand_distribution[left_upper - 1] += left_scale * quantize_distribution[j]
                    if right_scale > 0:
                        expand_distribution[right_lower] += right_scale * quantize_distribution[j]
                    expand_distribution[left_upper:right_lower] += quantize_distribution[j]

                # process upper edge
                start = threshold - num_per_bin
                left_upper = int(np.ceil(start))
                left_scale = left_upper - start
                if left_scale > 0:
                    expand_distribution[left_upper - 1] += left_scale * quantize_distribution[target_bin - 1]
                expand_distribution[left_upper:threshold] += quantize_distribution[target_bin - 1]

                kl_divergence = compute_kl_divergence(clip_distribution, expand_distribution)
                if kl_divergence < min_kl_divergence:
                    min_kl_divergence = kl_divergence
                    target_threshold = threshold

            self.quant_blob_stats[idx].threshold = (target_threshold + 0.5) * self.quant_blob_stats[idx].absmax / num_histogram_bins
            self.bottom_blob_scales[idx] = 127 / self.quant_blob_stats[idx].threshold
            print("%-10s : %-10f"%(self.layers()[self.conv_layers[idx]].name, self.quant_blob_stats[idx].threshold))

    def print_quant_info(self):
        conv_bottom_blob_count = len(self.conv_bottom_blobs)
        for index in range(conv_bottom_blob_count):
            stat = self.quant_blob_stats[index]
            scale = self.bottom_blob_scales[index]
            print("%-40s : max = %-15f threshold = %-15f scale = %-15f" % (
                self.layers()[self.conv_layers[index]].name,
                stat.absmax,
                stat.threshold,
                scale,
            ))

    def save_table(self, table_path):
        with open(table_path, 'w') as fpW:
            conv_layer_count = len(self.conv_layers)
            for index in range(conv_layer_count):
                fpW.write("%s_param_0 " % self.layers()[self.conv_layers[index]].name)
                for w_scale in self.weight_scales[index]:
                    fpW.write("%f " % (w_scale))
                fpW.write("\n")

            conv_bottom_blob_count = len(self.conv_bottom_blobs)
            for index in range(conv_bottom_blob_count):
                fpW.write("%s " % self.layers()[self.conv_layers[index]].name)
                fpW.write("%f\n" % (self.bottom_blob_scales[index]))


if __name__ == '__main__':
    opt = ncnn.Option()
    opt.num_threads = 1
    opt.use_fp16_packed = False
    opt.use_fp16_storage = False
    opt.use_fp16_arithmetic = False

    input_param = "best_320_opt.param"
    input_bin = "best_320_opt.bin"
    norm = [0.003921, 0.003921, 0.003921]
    shape = [320, 320, 3]

    # input_param = "lenet.param"
    # input_bin = "lenet.bin"
    # norm = [0.003921, 0.003921, 0.003921]
    # shape = [32, 32, 3]

    img_list = ["img_list.txt"]
    net = QuantNet(input_param, input_bin, img_list, type_to_pixels=[ncnn.Mat.PixelType.PIXEL_BGR], norms=[norm], shapes=[shape])
    net.opt = opt
    net.init()
    net.quantize_KL()
    net.print_quant_info()
    net.save_table(input_param.replace("param", "table"))