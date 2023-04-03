#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <ros/ros.h>
#include "std_msgs/Float32.h"

#include "logger.cpp"

class SimpleNN
{

public:
    SimpleNN(
        const std::string engine_file,
        const std::string input_tensor_name,
        const std::string output_tensor_name);

    bool infer(const int input);

    bool deserializeEngineFromFile();

private:

    bool processInput(
        const samplesCommon::BufferManager &buffers,
        const int input);

    bool verifyOutput(
        const samplesCommon::BufferManager &buffers);
    
    void callback(const std_msgs::Float32::ConstPtr& float_msg);

    const std::string engine_file_;

    const std::string input_tensor_name_;

    const std::string output_tensor_name_;

    nvinfer1::Dims input_dims_;

    nvinfer1::Dims output_dims_;

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;

    ros::NodeHandle nh_;

    ros::NodeHandle nh_private_;

    ros::Subscriber sub_;

    ros::Publisher pub_;

    float cur_output_;

    std::shared_ptr<samplesCommon::BufferManager> buffers_;

    samplesCommon::SampleUniquePtr<nvinfer1::IExecutionContext> context_;
};
