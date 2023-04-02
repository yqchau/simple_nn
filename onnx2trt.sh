#! /bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=./models/simple_nn2.onnx --saveEngine=./models/simple_nn2.engine
