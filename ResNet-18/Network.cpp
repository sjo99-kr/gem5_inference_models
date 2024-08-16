#include "Network.h"
#include <vector>
#include <iostream>
#include <string>
#include <gem5/m5ops.h>
#define GEM5
//#include <gem5/m5ops.h>
//#define GEM5

// 112x112x32 -> input data
using namespace std;

Network::Network()
{

	cout << "Initializing Network..." << endl;

  // if you want to use RGB image data, you have to add conv2d layer (7x7 kernel_size)

  
    m_Layers_ds1_1 = new Layers_Ds(64, 64, 56, 3, 1, 1 ); // input : 56x56x64, padding : 1, stride : 1 kernel_size : 3 output: 56x56x64

    m_Layers_ds1_2 = new Layers_Ds(64, 64, 56, 3, 1, 1); // input : 56x56x64, padding : 1, stride : 1, kernel_size : 3 output 56x56x64

    m_Layers_ds2_1 = new Layers_Ds(64, 128, 56, 3, 2 ,1); // input: 56x56x64, padding 1, stride: 2 kernel_size : 3 output : 28x28x128

    m_Layers_ds2_2 = new Layers_Ds(128, 128, 28, 3, 1, 1); // input 28x28x128, padding : 1 stride: 1 kernel_size : 3 output: 28x28x128
    
    m_Layers_ds3_1 = new Layers_Ds(128, 256, 28, 3, 2, 1); // input : 28x28x128, padding : 1 stride : 2 kernel_size : 3, output 14x14x256

    m_Layers_ds3_2 = new Layers_Ds(256, 256, 14, 3, 1, 1); // input : 14x14x256 padding : 1, stride 1 kernel_size 3 output 14x14x256

    m_Layers_ds4_1 = new Layers_Ds(256, 512, 14, 3, 2, 1); //input : 14x14x256 padding : 1, stride : 2 kernel_size :3 output 7x7x512

    m_Layers_ds4_2 = new Layers_Ds(512, 512, 7, 3, 1, 1); //input : 7x7x512 padding : 1 stride: 1 kernel_size : 3, output : 7x7x512


    m_Poollayer6 = new GlobalPoolLayer(512, 7);


    m_Fclayer7 = new FcLayer(512, 12);

    m_Sigmoidlayer8 = new SigmoidLayer(12);

    cout << "Initializing Done..." << endl;
    cout << endl;

}

/*
Network::~Network()
{
	delete m_Readdata;
    delete m_Layers_bn;
    delete m_Layers_ds1;
    delete m_Layers_ds2_1;
    delete m_Layers_ds2_2;
    delete m_Layers_ds3_1;
    delete m_Layers_ds3_2;
    delete m_Layers_ds4_1;
    delete m_Layers_ds4_2;
    delete m_Layers_ds5_1;
    delete m_Layers_ds5_2;
    delete m_Layers_ds5_3;
    delete m_Layers_ds5_4;
    delete m_Layers_ds5_5;
    delete m_Layers_ds5_6;
    delete m_Layers_ds6;
    delete m_Poollayer6;
    //delete m_Convlayer7;
    delete m_Fclayer7;
    delete m_Sigmoidlayer8;
}
*/

void Network::Forward(const char *pcName)
{

    const int size = 56 * 56 * 64;
    float* inputData = new float[size];
    for (int i =0; i<size; ++i){
    	inputData[i] = static_cast<float>(i);
    }
    std::cout << "layer 1 " << std::endl;
    std::cout << "Set done for input Data _Resnet 56x56x64 version" << std::endl;

#ifdef GEM5
        m5_dump_reset_stats(0,0);
#endif
    m_Layers_ds1_1->forward(inputData);
    std::cout << "layer 1" << std::endl;
    m_Layers_ds1_2->forward(m_Layers_ds1_1->GetOutput());
    std::cout << "layer 2" << std::endl;
    m_Layers_ds2_1->forward(m_Layers_ds1_2->GetOutput());
    std::cout << "layer 3" << std::endl;
    m_Layers_ds2_2->forward(m_Layers_ds2_1->GetOutput());
    std::cout << "layer 4" << std::endl;
    m_Layers_ds3_1->forward(m_Layers_ds2_2->GetOutput());
    std::cout << "layer 5" << std::endl;
    m_Layers_ds3_2->forward(m_Layers_ds3_1->GetOutput());
    std::cout << "layer 6" << std::endl;
    
    m_Layers_ds4_1->forward(m_Layers_ds3_2->GetOutput());
    std::cout << "layer 7" << std::endl;
    m_Layers_ds4_2->forward(m_Layers_ds4_1->GetOutput());
    std::cout << "layer 8" << std::endl;
        std::cout << "layer 10" << std::endl;
    m_Poollayer6->forward(m_Layers_ds4_2->GetOutput());
    std::cout << "layer 11" << std::endl;
    //m_Convlayer7->forward(m_Poollayer6->GetOutput());
    m_Fclayer7->forward(m_Poollayer6->GetOutput());
    std::cout << "layer 12" << std::endl;
    //m_Sigmoidlayer8->forward(m_Convlayer7->GetOutput());
    m_Sigmoidlayer8->forward(m_Fclayer7->GetOutput());
#ifdef GEM5
        m5_dump_reset_stats(0,0);
#endif
    float *pfOutput = m_Sigmoidlayer8->GetOutput();


}
