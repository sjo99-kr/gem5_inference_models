#define _CRT_SECURE_NO_WARNINGS

#include "fcLayer.h"
#include <cmath>
#include <cstring> 
//#include <ctime>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;


FcLayer::FcLayer(int nInputSize, int nOutputSize) :
    m_nInputSize(nInputSize), m_nOutputSize(nOutputSize)
    //m_pcWname(pcWname), m_pcBname(pcBname), m_nInputSize(nInputSize), m_nOutputSize(nOutputSize)
{
    m_nWeightSize = m_nInputSize * m_nOutputSize;
    m_pfWeight = new float[m_nWeightSize];
    m_pfBias = new float[m_nOutputSize];
    m_pfOutput = new float[m_nOutputSize];
    ReadFcWb();
}

FcLayer::~FcLayer()
{
    delete[] m_pfOutput;
    delete[] m_pfWeight;
    delete[] m_pfBias;
}

void FcLayer::forward(float *pfInput)
{

    for(int i = 0; i < m_nOutputSize; i++)
    {
        float fSum = 0.0;
        int weight_index;
        for(int j = 0; j < m_nInputSize; j++)
        {
            weight_index = i * m_nInputSize + j;
            fSum += m_pfWeight[weight_index] * pfInput[j];
        }
        fSum += m_pfBias[i];


        m_pfOutput[i] = fSum;
    }
}

void FcLayer::ReadFcWb()
{
    int nWsize, nBsize, nWreadsize, nBreadsize;
    for(int i =0; i<m_nWeightSize; ++i){
    	m_pfWeight[i] = static_cast<float>(i+0.125);
    }
    for(int i =0; i< m_nOutputSize; ++i){
    	m_pfBias[i] = static_cast<float>(i+0.10254);
    }

   
}

float *FcLayer::GetOutput()
{
    return m_pfOutput;

}

int FcLayer::GetOutputSize()
{
	return m_nOutputSize;
}
