#ifndef LAYERS_BOTTLENECK_H
#define LAYERS_BOTTLENECK_H

#include "readdata.h"
#include "convLayer.h"
#include "globalpoolLayer.h"
#include "batchnormalLayer.h"
#include "reluLayer.h"
#include "sigmoidLayer.h"
#include <vector>
#include <iostream>
#include <string>

class layer_bottleneck
{
public:
    layer_bottleneck(int nInputNum, int nOutputNum, int nExpansionFactor, int nInputWidth, int nKernelWidth, int nPad, int nStride);
    ~layer_bottleneck();
    void forward(float *pfInput);
    float *GetOutput();
    int nStride;
private:
    float *m_nOutput;
    int nOutputNum;
    int nInputWidth;

    int nExpansionFactor;
    ConvLayer *m_ConvlayerSep2, *m_ConvlayerSep1, *m_depth_wise_conv;
    BatchNormalLayer *m_ConvSepBn1, *m_ConvSepBn2, *m_depth_wise_bn;
    ReluLayer *m_RelulayerSep1, *m_depth_wise_relu;
    

};


#endif
