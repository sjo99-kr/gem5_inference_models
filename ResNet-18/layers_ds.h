#ifndef LAYERS_DS_H
#define LAYERS_DS_H

#include "readdata.h"
#include "convLayer.h"
#include "globalpoolLayer.h"
#include "batchnormalLayer.h"
#include "reluLayer.h"
#include "sigmoidLayer.h"
#include <vector>
#include <iostream>
#include <string>

class Layers_Ds
{
public:
    Layers_Ds(int nInputNum, int nOutputNum, int nInputWidth, int nKernelWidth, int nPad, int nStride);
    ~Layers_Ds();
    void forward(float *pfInput);
    float *GetOutput();

private:
    float *m_nOutput;
    int nOutputNum;
    int nInputWidth;
    int nStride;
    ConvLayer *m_ConvlayerSep2, *m_ConvlayerSep1, *m_shortCut_conv;
    BatchNormalLayer *m_ConvSepBn1, *m_ConvSepBn2, *m_shortCut_bn;
    ReluLayer *m_RelulayerSep1, *m_RelulayerSep2;
    

};


#endif
