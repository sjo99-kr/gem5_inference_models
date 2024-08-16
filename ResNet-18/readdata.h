#include "readdata.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <fstream>

using namespace std;

ReadData::ReadData(const char *pcMean, int nInputWidth, int nInputHeight, int nInputChannel)
    : m_nInputWidth(nInputWidth), m_nInputHeight(nInputHeight), m_nInputChannel(nInputChannel)
{
    m_nImageSize = nInputWidth * nInputHeight;
    m_nInputSize = nInputWidth * nInputHeight * nInputChannel;
    m_pfInputData = new float[m_nInputSize];
    m_pfMean = new float[m_nInputSize];
    ReadMean(pcMean);
}

ReadData::~ReadData()
{
    delete[] m_pfInputData;
    delete[] m_pfMean;
}

float *ReadData::ReadInput(const char *pcName)
{
    cout << "Reading Picture: " << pcName << "..." << endl;

    cv::Mat pSrcImage = cv::imread(pcName, cv::IMREAD_UNCHANGED);
    if (pSrcImage.empty()) {
        cerr << "Error: Could not load image!" << endl;
        return nullptr;  // 또는 예외 처리
    }
    std::cout << "image input access 1 \n" <<std::endl;
    cv::Mat pDstImage;
    cv::resize(pSrcImage, pDstImage, cv::Size(m_nInputWidth, m_nInputHeight), 0, 0, cv::INTER_LINEAR);
    std::cout << "image input access 2 \n" <<std::endl;
    // 데이터의 채널 수를 확인하고 데이터를 복사합니다.
    int channels = pDstImage.channels();
    uchar* pucData = pDstImage.data;
    int nStep = pDstImage.step[0]; // row step size

    std::cout << "image input access 3 \n" <<std::endl;
    std::cout << pDstImage.rows << std::endl; // 224
    std::cout << pDstImage.cols << std::endl; // 224
    std::cout << m_nInputWidth << std::endl; //  224
    std::cout << channels << std::endl;  // 4




    std::cout << "image input access 4 \n" << std::endl;
    int nOutputIndex = 0;

    for (int i = 0; i < pDstImage.rows; i++)
    {
        for (int j = 0; j < pDstImage.cols; j++)
        {
            nOutputIndex = i * m_nInputWidth + j;
            for (int c = 0; c < channels-1; c++)
            {
                m_pfInputData[nOutputIndex + c * m_nImageSize] = (float)pucData[i * nStep + j * channels + c] - m_pfMean[nOutputIndex + c * m_nImageSize];
		
            }
        }
    }
    

    cout << "Reading Picture Done..." << endl;

    return m_pfInputData;
}

void ReadData::ReadMean(const char *pcMean)
{
    ifstream file(pcMean, ios::binary);
    if (!file) {
        cerr << "Error: Could not open mean file!" << endl;
        return;  // 또는 예외 처리
    }

    file.read(reinterpret_cast<char*>(m_pfMean), m_nInputSize * sizeof(float));
    file.close();
}
