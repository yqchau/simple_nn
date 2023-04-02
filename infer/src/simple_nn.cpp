#include "simple_nn.h"

SimpleNN::SimpleNN(
    const std::string engineFile,
    const std::string inputTensorName,
    const std::string outputTensorName) : engine_file_{engineFile},
                                          input_tensor_name_{inputTensorName},
                                          output_tensor_name_{outputTensorName}
{
    std::cout << "Loading parameters.." << std::endl;
    std::cout << "EngineFile: " << engine_file_ << std::endl;
    std::cout << "InputTensorName: " << input_tensor_name_ << std::endl;
    std::cout << "OutputTensorName: " << output_tensor_name_ << std::endl;
    std::cout << "Done!\n"
              << std::endl;
}

bool SimpleNN::deserializeEngineFromFile()
{
    // read the file into a buffer
    std::ifstream engine_file(engine_file_, std::ios::binary);
    if (!engine_file)
    {
        return false;
    }

    engine_file.seekg(0, engine_file.end);
    const int engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    std::unique_ptr<char[]> engineBuffer(new char[engine_size]);
    if (!engineBuffer)
    {
        return false;
    }

    if (!engine_file.read(engineBuffer.get(), engine_size))
    {
        return false;
    }

    // deserialize the engine
    samplesCommon::SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engineBuffer.get(), engine_size), samplesCommon::InferDeleter());
    if (!engine_)
    {
        return false;
    }

    ASSERT(engine_->getNbBindings() == 2);
    ASSERT(engine_->bindingIsInput(0) && !engine_->bindingIsInput(1));

    input_dims_ = engine_->getBindingDimensions(0);
    std::cout << "input_dims: " << input_dims_ << std::endl;
    ASSERT(input_dims_.nbDims == 2);

    output_dims_ = engine_->getBindingDimensions(1);
    std::cout << "output_dims: " << output_dims_ << std::endl;
    ASSERT(output_dims_.nbDims == 2);

    return true;
}

bool SimpleNN::infer(const int input)
{
    samplesCommon::BufferManager buffers(engine_);

    auto context = samplesCommon::SampleUniquePtr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());

    if (!context)
    {
        return false;
    }

    // ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, input))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

bool SimpleNN::verifyOutput(const samplesCommon::BufferManager &buffers)
{
    const int output_size = output_dims_.d[1];
    float *output = static_cast<float *>(
        buffers.getHostBuffer(
            output_tensor_name_));

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < output_size; i++)
    {
        sample::gLogInfo << "Output: " << output[i] << std::endl;
    }

    return true;
}

bool SimpleNN::processInput(const samplesCommon::BufferManager &buffers, const int input)
{
    const int input_dim = input_dims_.d[1];
    float inputData = input;

    float *host_data_buffer = static_cast<float *>(
        buffers.getHostBuffer(input_tensor_name_));

    for (int i = 0; i < input_dim; i++)
    {
        host_data_buffer[i] = inputData;
    }

    std::cout << "host_data_buffer: " << host_data_buffer[input_dim - 1] << std::endl;

    return true;
}

int main(int argc, char **argv)
{

    assert(argc == 3);

    const int input{std::stoi(argv[2])};

    const std::string engine_file{argv[1]};

    const std::string input_tensor_name = "input";

    const std::string output_tensor_name = "3";

    SimpleNN model(
        engine_file,
        input_tensor_name,
        output_tensor_name);

    model.deserializeEngineFromFile();

    model.infer(input);

    return 0;
}