#ifndef POOLAYER_H
#define POOLAYER_H

class AveragePoolLayer
{
public:
    AveragePoolLayer(int nOutputNum, int nInputWidth);
    ~AveragePoolLayer();
    void forward(float *pfInput);
    float *GetOutput();

private:
    int m_nOutputNum, m_nPoolWidth, m_nInputWidth;
    int m_nPoolSize, m_nOutputWidth, m_nInputSize, m_nOutputSize;
    
    float *m_pfOutput;
};

#endif
