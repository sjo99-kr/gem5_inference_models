#define _CRT_SECURE_NO_WARNINGS

#include "reluLayer.h"
#include <iostream>

using namespace std;

ReluLayer::ReluLayer(int nInputSize) : m_nInputSize(nInputSize)
{
    m_pfOutput = new float[m_nInputSize];
}

ReluLayer::~ReluLayer()
{
    delete[] m_pfOutput;
}

void ReluLayer::forward(float *pfInput)
{
    for (int i = 0; i < m_nInputSize; i++)
    {
        if (pfInput[i] > 0)
            if(pfInput[i] > 6){
            	// ReLU 6
              // if you wanna use ReLU, you have to change this part
            	m_pfOutput[i] = 6;
            }
            else {
            	m_pfOutput[i] = pfInput[i];
            }
        else
            m_pfOutput[i] = 0;
    }
}

float *ReluLayer::GetOutput()
{
    return m_pfOutput;
}
