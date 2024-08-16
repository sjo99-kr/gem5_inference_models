#ifndef FCLAYER_H
#define FCLAYER_H
#include <string>

class FcLayer
{
public:
    FcLayer(int nInputSize, int nOutputSize);
    ~FcLayer();
    void forward(float *pfInput);
    float *GetOutput();
    int GetOutputSize();
    void ReadFcWb();
    
private:
    int m_nInputSize, m_nOutputSize, m_nWeightSize, m_nRelubool;
    float *m_pfWeight, *m_pfBias, *m_pfOutput;
};

#endif
