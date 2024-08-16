#define _CRT_SECURE_NO_WARNINGS

#include "convLayer.h"
#include <fstream>
#include <cassert>
#include <iostream>

using namespace std;

//////////////////////////////////////////////////////////////////// set for MobileNet-V2  ////////////////////////////////////////////////
ConvLayer::ConvLayer(int nInputNum, int nOutputNum, int nInputWidth, int nKernelWidth, int nPad, int nStride, int group) :
    m_nInputNum(nInputNum), m_nOutputNum(nOutputNum), m_nInputWidth(nInputWidth), m_nGroup(group),
    m_nKernelWidth(nKernelWidth), m_nPad(nPad), m_nStride(nStride)
{
    m_nKernelSize = m_nKernelWidth * m_nKernelWidth; // set kernel (filter) size , maybe 3x3 or 1x1
    m_nInputSize = m_nInputWidth * m_nInputWidth; // set input Featuremap size
    m_nInputPadWidth = m_nInputWidth + 2* m_nPad; 
    m_nInputPadSize = m_nInputPadWidth * m_nInputPadWidth; // set input Featuremap size with padding, if same_padding, the value will be original_size + 2  
    m_nOutputWidth = int((m_nInputPadWidth- m_nKernelWidth) / m_nStride + 1); // calculation output featuremap width ((input_width - filter_size + 2*padding)/ stride + 1)
    m_nOutputSize = m_nOutputWidth * m_nOutputWidth;


    m_nInputGroupNum = m_nInputNum / m_nGroup; // if depth-wise convolution, m_nInputGroupNum = 1
    m_nOutputGroupNum = m_nOutputNum / m_nGroup; // if depth-wise convolution, m_nOutputGroupNum = 1

    m_pfWeight = new float[m_nOutputNum * m_nInputNum * m_nKernelSize]; // depth wise -> m_nInputGroupNum = 1, else m_nInputGroupNum = m_nInputNUm
	
    // set weight, we can set weights by parameter files, but in gem5, Our goal is to analyze the  Inference time breakdown based on Structure of CNN Model
  for(int i =0; i<m_nOutputNum * m_nInputNum * m_nKernelSize; ++i){
         m_pfWeight[i] = static_cast<float>(i);
    }
    m_pfBias = new float[m_nOutputNum];
    for(int i =0; i<m_nOutputNum; ++i){	
        m_pfBias[i] = static_cast<float>(i);
    }
    std::cout << "conv setting on " << std::endl;
}



void ConvLayer::forward(float *pfInput)
{
    Addpad(pfInput);
    m_pfOutput = new float[m_nOutputNum * m_nOutputSize];

    for (int g = 0; g < m_nGroup ; g++)
    {
        for (int nOutmapIndex = 0; nOutmapIndex < m_nOutputGroupNum; nOutmapIndex++)
        {
            for (int i = 0; i < m_nOutputWidth; i++)
            {
                for (int j = 0; j < m_nOutputWidth; j++)
                {
                    float fSum = 0;
                    int nInputIndex, nOutputIndex, nKernelIndex, nInputIndexStart, nKernelStart;
                    nOutputIndex = g * m_nInputGroupNum * m_nOutputSize + nOutmapIndex * m_nOutputSize + i * m_nOutputWidth + j;
                    for (int k = 0; k < m_nInputGroupNum; k++)
                    {
                        nInputIndexStart = g * m_nInputGroupNum * m_nInputPadSize + k * m_nInputPadSize + (i * m_nStride) * m_nInputPadWidth + (j * m_nStride);
                        nKernelStart = g * m_nOutputGroupNum * m_nKernelSize + nOutmapIndex * m_nInputGroupNum * m_nKernelSize + k * m_nKernelSize;
                        for (int m = 0; m < m_nKernelWidth; m++)
                        {
                            for (int n = 0; n < m_nKernelWidth; n++)
                            {
                                nKernelIndex = nKernelStart + m * m_nKernelWidth + n;
                                nInputIndex = nInputIndexStart + m * m_nInputPadWidth + n;
                                fSum += m_pfInputPad[nInputIndex] * m_pfWeight[nKernelIndex];
                            }
                        }
                    }
                    if (true) // if bias parameter is not existed , then you can set false
                        fSum += m_pfBias[nOutmapIndex];

                    m_pfOutput[nOutputIndex] = fSum;
                }
            }
        }              
    }
}


float *ConvLayer::GetOutput()
{
    return m_pfOutput;
}

void ConvLayer::Addpad(float *pfInput)
{
    m_pfInputPad = new float[m_nInputPadWidth *m_nInputPadWidth * m_nInputNum];
    for (int m = 0; m < m_nInputNum; m++)
	{
		for (int i = 0; i < m_nInputPadWidth; i++)
		{
			for (int j = 0; j < m_nInputPadWidth; j++)
			{
                if ((i < m_nPad) || (i >= m_nInputPadWidth - m_nPad))
                {
                    m_pfInputPad[m * m_nInputPadSize + i * m_nInputPadWidth + j] = 0;
                }
                else if ((j < m_nPad) || (j >= m_nInputPadWidth - m_nPad))
                {
                    m_pfInputPad[m * m_nInputPadSize + i * m_nInputPadWidth + j] = 0;
                }
                else
                {
                    m_pfInputPad[m * m_nInputPadSize + i * m_nInputPadWidth + j] = pfInput[m * m_nInputSize + (i - m_nPad) * m_nInputWidth + (j - m_nPad)];
                }
			}
		}
	}
}

int ConvLayer::GetOutputSize()
{
    return m_nOutputNum * m_nOutputSize;
}
