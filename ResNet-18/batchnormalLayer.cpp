#define _CRT_SECURE_NO_WARNINGS

#include "batchnormalLayer.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <cassert>
#include <cmath> //vs²»ÐèÒªÕâ¸ö£¬g++ÐèÒªÕâ¸ö

using namespace std;

BatchNormalLayer::BatchNormalLayer(int nInputNum, int nInputWidth) :
    m_nInputNum(nInputNum), m_nInputWidth(nInputWidth)
{
    m_nInputSize = m_nInputWidth * m_nInputWidth;
    m_pfOutput = new float[m_nInputNum * m_nInputSize];
    m_pfMean = new float[m_nInputNum];
    m_pfVar = new float[m_nInputNum];
    m_pfFiller = new float[m_nInputNum];
    m_pfBias = new float[m_nInputNum];
    ReadParam();
}

BatchNormalLayer::~BatchNormalLayer()
{
    delete[] m_pfOutput;
    delete[] m_pfMean;
    delete[] m_pfVar;
    delete[] m_pfFiller;
    delete[] m_pfBias;
}

void BatchNormalLayer::forward(float *pfInput) 
{
    for (int i = 0; i < m_nInputNum; i++)
    {
        for (int j = 0; j < m_nInputSize; j++)
        {
            int nOutputIndex = i * m_nInputSize + j;

            m_pfOutput[nOutputIndex] = m_pfFiller[i] * ((pfInput[nOutputIndex] - m_pfMean[i])
                / sqrt(m_pfVar[i] + 1e-5)) + m_pfBias[i];
        }
    }
}

void BatchNormalLayer::ReadParam()
{
    int nMsize, nVsize, nFsize, nBsize, nMreadsize, nVreadsize, nFreadsize, nBreadsize;



    nMsize = m_nInputNum;
    nVsize = m_nInputNum;
    nFsize = m_nInputNum;
    nBsize = m_nInputNum;
    for(int i =0; i<m_nInputNum; ++i){
    
    	m_pfFiller[i] = static_cast<float>(i+0.1);
    	m_pfMean[i] = static_cast<float>(i+0.01);
    	m_pfVar[i] = static_cast<float>(i+0.001);
    	m_pfBias[i] = static_cast<float>(i+0.12);
    	
    }

    
}

float *BatchNormalLayer::GetOutput()
{
    return m_pfOutput;
}

int BatchNormalLayer::GetOutputSize()
{
    return m_nInputNum * m_nInputSize;
}
