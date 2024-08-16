#include "utils.cpp"
#include "fcLayer.cpp"
#include "globalpoolLayer.cpp"
#include "Network.cpp"
#include "layer_bottleneck.cpp"
#include "convLayer.cpp"
#include "reluLayer.cpp"
#include "readdata.cpp"
#include "batchnormalLayer.cpp"

#include "sigmoidLayer.cpp"

int main()
{
    std::cout<< "STRAT MOBILENET V2 "<< std::endl;
    Network network;
    network.Forward("ImageFile");
    std::cout << "Inference Done" << std::endl;
    return 0;
}
