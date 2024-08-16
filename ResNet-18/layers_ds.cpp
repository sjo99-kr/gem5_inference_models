#include "layers_ds.h"



Layers_Ds::Layers_Ds(int nInputNum, int nOutputNum, int nInputWidth, int nkernel_size, int nStride, int nPad) :
	nOutputNum(nOutputNum), nInputWidth(nInputWidth), nStride(nStride)
{

    m_ConvlayerSep1 = new ConvLayer(nInputNum, nOutputNum, nInputWidth, nkernel_size, nPad, nStride, 1); // padding: 1 , stride: nStride

    
    int stride_width = int((nInputWidth + 2*nPad - nkernel_size) / nStride + 1);
    
    m_ConvSepBn1 = new BatchNormalLayer(nOutputNum, (stride_width));

    m_RelulayerSep1 = new ReluLayer(nOutputNum * stride_width * stride_width);


    m_shortCut_conv = new ConvLayer(nInputNum, nOutputNum, nInputWidth, 1, nPad, nStride, 1);

    m_shortCut_bn = new BatchNormalLayer(nOutputNum, stride_width);
    

    m_ConvlayerSep2 = new ConvLayer(nInputNum, nOutputNum, stride_width, nkernel_size, nPad, 1, 1); // padding: 1 , stride: 1

    m_ConvSepBn2 = new BatchNormalLayer(nOutputNum, stride_width);

    m_RelulayerSep2 = new ReluLayer(nOutputNum * stride_width * stride_width);

}



void Layers_Ds::forward(float *pfInput)
{
    std::cout<<"conv1" <<std::endl;
    m_ConvlayerSep1->forward(pfInput);
    std::cout<<"bn1" <<std::endl;
    m_ConvSepBn1->forward(m_ConvlayerSep1->GetOutput());
    std::cout<<"relu1" <<std::endl;
    m_RelulayerSep1->forward(m_ConvSepBn1->GetOutput());
////// Identity setting  /////////////////////////////////
    std::cout<<"conv-short 1" <<std::endl;
    m_shortCut_conv->forward(pfInput);
    std::cout<<"bn-short 1" <<std::endl;
    m_shortCut_bn->forward(m_shortCut_conv->GetOutput());

/////Short Cut setting////////////////////////////////////
    std::cout<<"conv 2" <<std::endl;
    m_ConvlayerSep2->forward(m_RelulayerSep1->GetOutput());
    std::cout<<"bn 2" <<std::endl;
    m_ConvSepBn2->forward(m_ConvlayerSep2->GetOutput());
    std::cout<<"relu 2" <<std::endl;
    m_RelulayerSep2->forward(m_ConvSepBn2->GetOutput());
    
    for(int i =0; i< nOutputNum * nInputWidth / nStride * nInputWidth / nStride; ++i){
    	m_RelulayerSep2->m_pfOutput[i] = m_RelulayerSep2->m_pfOutput[i] + m_shortCut_bn->m_pfOutput[i];
    }
}

float *Layers_Ds::GetOutput()
{
    return m_RelulayerSep2->GetOutput();
}
