#ifndef CONVLAYER_H
#define CONVLAYER_H
#include <string>

class ConvLayer
{
public:
    ConvLayer(int nInputNum, int nOutputNum, int nInputWidth, int nKernelWidth, int nPad, int nStride, int group);

    void forward(float *pfInput);
    float *GetOutput();
    void Addpad(float *pfInput);
    int GetOutputSize();

private:
    int m_nInputNum, m_nOutputNum, m_nInputWidth, m_nKernelWidth, m_nPad, m_nStride, m_nGroup, m_nInputGroupNum, m_nOutputGroupNum;
    float *m_pfInputPad, *m_pfOutput;
    float *m_pfWeight;
    float *m_pfBias;
    int m_nInputPadWidth, m_nOutputWidth;
    int m_nKernelSize, m_nInputSize, m_nInputPadSize, m_nOutputSize;
};
#endif
