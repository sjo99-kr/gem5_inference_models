#ifndef NETWORK_H
#define NETWORK_H

#include "readdata.h"
#include "convLayer.h"
#include "globalpoolLayer.h"
#include "batchnormalLayer.h"
#include "reluLayer.h"
#include "fcLayer.h"
#include "sigmoidLayer.h"
#include "layer_bottleneck.h"
#include <vector>
#include <iostream>
#include <string>

using namespace std;

class Network
{
public:
    Network();
    void Forward(const char *pcName);
    

private:
    float *m_pfOutput;
    float m_dAccuracy;
    int m_nPrediction;
    vector <const char *> m_vcClass;
    const char *m_cClass[10];
    ReadData *m_Readdata;
    layer_bottleneck *m_Layers_ds1_1, *m_Layers_ds2_1, *m_Layers_ds2_2, *m_Layers_ds3_1, *m_Layers_ds3_2, *m_Layers_ds3_3, *m_Layers_ds4_1, *m_Layers_ds4_2, *m_Layers_ds4_3, *m_Layers_ds4_4,
        *m_Layers_ds5_1, *m_Layers_ds5_2, *m_Layers_ds5_3, *m_Layers_ds6_1, *m_Layers_ds6_2, *m_Layers_ds6_3, *m_Layers_ds7_1, *m_Layers_ds8_1;



    FcLayer *m_Fclayer10;

    GlobalPoolLayer *m_Poollayer9;

    SigmoidLayer *m_Sigmoidlayer11;
};


#endif
