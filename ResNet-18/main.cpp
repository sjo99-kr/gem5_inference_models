#include "utils.cpp"
#include "fcLayer.cpp"
#include "globalpoolLayer.cpp"
#include "Network.cpp"
#include "layers_ds.cpp"
#include "convLayer.cpp"
#include "reluLayer.cpp"
#include "readdata.cpp"
#include "batchnormalLayer.cpp"

#include "sigmoidLayer.cpp"

int main()
{
    std::cout << "Resnet-18 Inference Start" <<std::endl;
    Network network;
    network.Forward("Image for Inference");
    std::cout << "Resnet-18 Inference Done" << std::endl;
    return 0;
}
