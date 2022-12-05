#!/bin/bash

ncnn_root=/home/sunnypc/dangxs/build/ncnn-20220729/build/tools && \
onnx_root=./train_result/20221201_tid2013_mobilenetv2 && \
cktp_name=epoch_best.pth.onnx && \
ncnn_param=${onnx_root}/${cktp_name}.param && \
ncnn_bin=${onnx_root}/${cktp_name}.bin && \
${ncnn_root}/onnx/onnx2ncnn ${onnx_root}/${cktp_name} ${ncnn_param} ${ncnn_bin} && \
echo 'onnx2ncnn is done.'


opt_ncnn_param=${onnx_root}/${cktp_name}.opt.param && \
opt_ncnn_bin=${onnx_root}/${cktp_name}.opt.bin && \
#fp32=0 fp16=1
flag=0 && \
${ncnn_root}/ncnnoptimize ${ncnn_param} ${ncnn_bin} ${opt_ncnn_param} ${opt_ncnn_bin} ${flag} && \
echo 'ncnnoptimize is done.'

opt_ncnn_param=${onnx_root}/${cktp_name}.opt.param && \
opt_ncnn_bin=${onnx_root}/${cktp_name}.opt.bin && \
opt_ncnn_param1=${onnx_root}/${cktp_name}.opt.fp16.param && \
opt_ncnn_bin1=${onnx_root}/${cktp_name}.opt.fp16.bin && \
#fp32=0 fp16=1
flag=1 && \
${ncnn_root}/ncnnoptimize ${opt_ncnn_param} ${opt_ncnn_bin} ${opt_ncnn_param1} ${opt_ncnn_bin1} ${flag} && \
echo 'ncnnoptimize is done.'