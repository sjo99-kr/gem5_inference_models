#ifndef RELULAYER_H
#define RELULAYER_H

class ReluLayer
{
public:
    ReluLayer(int nInputSize);
    ~ReluLayer();
    void forward(float *pfInput);
    float *GetOutput();
    float *m_pfOutput;
private:
    int m_nInputSize;

};

#endif
