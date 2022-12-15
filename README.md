# NCNN模型量化流程(根据TensorRT的PPT实现，源码见$NCNNSRCROOT/tools/quantize/ncnn2table.cpp)

	本项目以NCNN模型(Yolov5s)`best_320_opt.bin`为例记录了模型量化过程


## Demo

```shell
# Python量化脚本，速度非常慢
python ncnn2table.py

#ncnn2table工具量化
ncnn2table best_320_opt.param best_320_opt.bin img_list.txt best_320_opt_ddd.table mean=[0,0,0] norm=[0.003921,0.003921,0.003921] shape=[320,320,3] pixel=BGR thread=1 method=kl
```

## ncnn_model_parse.py

该文件主要用于解析ncnn模型的权重，目前只测试了本项目中的演示模型`best_320_opt`，仅供学习参考

## 模型量化流程(基与KL散度)

###  1.初始化相关容器(QuantNet::init())
- quant_blob_stats
- weight_scales
- bottom_blob_scales
......

### 2.基于KL散度(相对熵)的量化

- 计算网络权重(Conv, ConvDepthWise, Inner)量化系数，逐通道(per-channel)量化
  - weight_scale\[i\] = 127 / absmax(每个通道权重或每组(针对ConvDepthWise)权重)

- 计算Conv, ConvDepthWise, Inner层(以下称：权重层)输入feature map的最大值absmax，逐Tensor(per-tensor)量化
  - 针对每个量化样本，计算权重层输入feature map的最大值
    input_feature_map_absmax_k_sample\[k\]\[i\] = absmax(input_feature_map\[i\]),
    input_feature_map_absmax\[i\] = absmax(input_feature_map_absmax_k_sample\[...\]\[i\])

- 计算Conv, ConvDepthWise, Inner层(以下称：权重层)输入feature map的直方图
  - 针对每个量化样本，根据上一步求出的input_feature_map_absmax\[i\]，将float32的输入激活值映射到[0, 2048)区间，并统计每个权重层输入feature map的直方图histogram\[k\]\[i\]\[bin\]。
  - histogram\[i\]\[bin\]=(histogram\[0\]\[i\]\[bin\] + histogram\[1\]\[i\]\[bin\] + ... + histogram\[N\]\[i\]\[bin\])

- 利用KL散度计算Conv, ConvDepthWise, Inner层(以下称：权重层)输入feature map的最佳阈值(量化系数)
  - 针对上一步求出的权重层输入feature map的直方图histogram\[i\]，先对其正则化
  - histogram_norm\[i\] = histogram\[i\]/sum(histogram\[...\])
  - for t in [128, 2048)，对于每个t计算clip_distribution和expand_distribution的kl散度(**重点**)
  - 求出kl散度最小时对应的t'，则T = (t' + 0.5)*input_feature_map_absmax\[i\] / 2048
  - input_feature_map_scale[i] = 127 / T


## 一些参考链接

ref: https://zhuanlan.zhihu.com/p/387072703

ref: https://github.com/Tencent/ncnn/blob/eaab441d3c/tools/quantize/ncnn2table.cpp

ref: https://github.com/Tencent/ncnn/wiki/operation-param-weight-table

ref: https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure

ref: https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf

ref: https://github.com/lutzroeder/netron/blob/main/source/ncnn.js
