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

  // If you want to original MobileNet-V2, you can use this code.
  // m_Layers_ds_start = new Conv2d(3, 32, 1, 224, 3, 2, 1), stride : 2, input : 224 x224x3 , padding : 1, kernel_size : 3, output 112x112x32


// bottleNeck 1, n = 1 
    m_Layers_ds1_1 = new layer_bottleneck(32, 16, 1, 112, 3, 1, 1); // input : 112x112x32, padding : 1, stride : 1 kernel_size : 3 output: 112x112x16

// BottleNeck 2,  n = 2
    m_Layers_ds2_1 = new layer_bottleneck(16, 24, 6, 112, 3, 2, 1); // input : 112x112x16, padding : 1, stride : 2, kernel_size : 3 output 56x56x24
    m_Layers_ds2_2 = new layer_bottleneck(24, 24, 6, 112, 3, 1 ,1); // input: 56x56x24, padding 1, stride: 1, kernel_size : 3 output : 56x56x24

// BottleNeck 3, n = 3
    m_Layers_ds3_1 = new layer_bottleneck(24, 32, 6, 56, 3, 2, 1); // input 56x56x24, padding : 1 stride: 2 kernel_size : 3 output: 28x28x32    
    m_Layers_ds3_2 = new layer_bottleneck(32, 32, 6, 28, 3, 1, 1); // input : 28x28x32, padding : 1 stride : 1 kernel_size : 3, output 28x28x32
    m_Layers_ds3_3 = new layer_bottleneck(32, 32, 6, 28, 3, 1, 1); // input : 28x28x32 padding : 1, stride 1 kernel_size 3 output 28x28x32


// BottleNeck 4, n = 4
    m_Layers_ds4_1 = new layer_bottleneck(32, 64, 6, 28, 3, 2, 1); //input : 28x28x32 padding : 1, stride : 2 kernel_size :3 output 14x14x64
    m_Layers_ds4_2 = new layer_bottleneck(64, 64, 6, 14, 3, 1, 1); //input : 14x14x64 padding : 1 stride: 1 kernel_size : 3, output : 14x14x64
    m_Layers_ds4_3 = new layer_bottleneck(64, 64, 6, 14, 3, 1, 1); //input : 14x14x64 padding : 1 stride: 1 kernel_size : 3, output : 14x14x64
    m_Layers_ds4_4 = new layer_bottleneck(64, 64, 6, 14, 3, 1, 1);  //input : 14x14x64 padding : 1 stride: 1 kernel_size : 3, output : 14x14x64
    
// BottleNeck 5, n =3
    m_Layers_ds5_1 = new layer_bottleneck(64, 96, 6, 14, 3, 1, 1); //input : 14x14x64 padding : 1, stride : 1 kernel_size :3 output 14x14x96
    m_Layers_ds5_2 = new layer_bottleneck(96, 96, 6, 14, 3, 1, 1); //input : 14x14x96 padding : 1 stride: 1 kernel_size : 3, output : 14x14x96
    m_Layers_ds5_3 = new layer_bottleneck(96, 96, 6, 14, 3, 1, 1); //input : 14x14x96 padding : 1 stride: 1 kernel_size : 3, output : 14x14x96

// BottleNeck 6, n = 3;
    m_Layers_ds6_1 = new layer_bottleneck(96, 160, 6, 14, 3, 2, 1); //input : 14x14x96 padding : 1, stride : 2 kernel_size :3 output 7x7x160
    m_Layers_ds6_2 = new layer_bottleneck(160, 160, 6, 7, 3, 1, 1); //input : 7x7x160 padding : 1 stride: 1 kernel_size : 3, output : 7x7x160
    m_Layers_ds6_3 = new layer_bottleneck(160, 160, 6, 7, 3, 1, 1); //input : 7x7x160 padding : 1 stride: 1 kernel_size : 3, output : 7x7x160
    
// BottleNeck 7, n =1;
    m_Layers_ds7_1 = new layer_bottleneck(160, 320, 6, 7, 3, 1, 1); // input : 7x7x160 padding : 1, stride : 1, kernel_size : 3, output : 7x7x320

// BottleNeck 8, n = 1;
    m_Layers_ds8_1 = new layer_bottleneck(320,1280, 6, 7, 3, 1, 1); // input 7x7x320 padding 1, stride 1, kernel_size :3 , output 7x7x1280
    
    
    m_Poollayer9 = new GlobalPoolLayer(1280, 7); // input 7x7x1280, output : 1280


    m_Fclayer10 = new FcLayer(1280, 12);

    m_Sigmoidlayer11 = new SigmoidLayer(12);

    cout << "Initializing Done..." << endl;
    cout << endl;

}



void Network::Forward(const char *pcName)
{


  // if you want to use rgb data or pcName arguments, you have to change the code.
    const int size = 112 * 112 * 32;
    float* inputData = new float[size];
    for (int i =0; i<size; ++i){
    	inputData[i] = static_cast<float>(i);
    }
    std::cout << "Set done for input Data _Resnet 112x112x32 version" << std::endl;

    std::cout << "layer 1" << std::endl;
#ifdef GEM5
        m5_dump_reset_stats(0,0);
#endif
    m_Layers_ds1_1->forward(inputData);
    
    std::cout << "layer 2" << std::endl;
    m_Layers_ds2_1->forward(m_Layers_ds1_1->GetOutput());
    m_Layers_ds2_2->forward(m_Layers_ds2_1->GetOutput());

    std::cout << "layer 3" << std::endl;
    m_Layers_ds3_1->forward(m_Layers_ds2_2->GetOutput());
    m_Layers_ds3_2->forward(m_Layers_ds3_1->GetOutput());
    m_Layers_ds3_3->forward(m_Layers_ds3_2->GetOutput());

    std::cout << "layer 4" << std::endl;
    m_Layers_ds4_1->forward(m_Layers_ds3_3->GetOutput());
    m_Layers_ds4_2->forward(m_Layers_ds4_1->GetOutput());
    m_Layers_ds4_3->forward(m_Layers_ds4_2->GetOutput());
    m_Layers_ds4_4->forward(m_Layers_ds4_3->GetOutput());
    
    std::cout << "layer 5" << std::endl;
    m_Layers_ds5_1->forward(m_Layers_ds4_4->GetOutput());
    m_Layers_ds5_2->forward(m_Layers_ds5_1->GetOutput());
    m_Layers_ds5_3->forward(m_Layers_ds5_2->GetOutput());

    std::cout << "layer 6" << std::endl;
    m_Layers_ds6_1->forward(m_Layers_ds5_3->GetOutput());
    m_Layers_ds6_2->forward(m_Layers_ds6_1->GetOutput());
    m_Layers_ds6_3->forward(m_Layers_ds6_2->GetOutput());

    std::cout << "layer 7" << std::endl;
    m_Layers_ds7_1->forward(m_Layers_ds6_3->GetOutput());

    std::cout << "layer 8" << std::endl;
    m_Layers_ds8_1->forward(m_Layers_ds7_1->GetOutput());
    
    std::cout << "Max Pooling Layer 9 "<< std::endl;
    m_Poollayer9->forward(m_Layers_ds8_1->GetOutput());
 
    std::cout << "FC Layer 10" << std::endl;
    m_Fclayer10->forward(m_Poollayer9->GetOutput());

    std::cout << "layer 12" << std::endl;
    m_Sigmoidlayer11->forward(m_Fclayer10->GetOutput());
    
#ifdef GEM5
        m5_dump_reset_stats(0,0);
#endif
    float *pfOutput = m_Sigmoidlayer11->GetOutput();


}
