#ifndef BATCHNORMAL_H
#define BATCHNORMAL_H

class BatchNormalLayer
{
public:
    BatchNormalLayer(int nInputNum, int nInputWidth);
    ~BatchNormalLayer();
    void forward(float *pfInput);
    float *GetOutput();
    int GetOutputSize();
    void ReadParam();
    float *m_pfOutput;
private:
    float m_fScale;
    int m_nInputNum, m_nInputWidth, m_nInputSize;
    float *m_pfMean, *m_pfVar, *m_pfFiller, *m_pfBias;

};

#endif
