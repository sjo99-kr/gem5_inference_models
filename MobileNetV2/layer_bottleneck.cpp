#include "layer_bottleneck.h"


// MobileNet v2 , 1x1Conv2d -> Relu6  -> 3x3 dwise -> Relu 6 -> linear 1xx1 conv2d  (BottleNeck structure)
// can divde structure based on Stride value

layer_bottleneck::layer_bottleneck(int nInputNum, int nOutputNum, int nExpansionFactor, int nInputWidth, int nkernel_size, int nStride, int nPad) :
	nOutputNum(nOutputNum), nInputWidth(nInputWidth), nStride(nStride), nExpansionFactor(nExpansionFactor)
{

    // MobileNet v2 (1x1 conv2d) (Expansion Factor)
    m_ConvlayerSep1 = new ConvLayer(nInputNum, nExpansionFactor * nInputNum, nInputWidth, 1, 0, 1, 1); // padding: 1 , stride: nStride

    // MobileNet v2 (Batch Normalization)
    m_ConvSepBn1 = new BatchNormalLayer(nExpansionFactor *nInputNum, nInputWidth);
    // MobileNet v2 (ReLU6)
    m_RelulayerSep1 = new ReluLayer(nExpansionFactor *nInputNum * nInputWidth * nInputWidth); // ReLU6


    // MobileNet v2 (3x3 dwise, stride = s), kernel_size = 3, padding = 1 (same), nstride = 2, group == depth-wise
    m_depth_wise_conv = new ConvLayer(nExpansionFactor * nInputNum, nExpansionFactor * nInputNum, nInputWidth, 3, nPad, nStride, nExpansionFactor * nInputNum);
    int stride_width = int((nInputWidth + 2 * nPad - 3)/ nStride + 1);

    // MobileNet v2 ( Batch Normalization)
    m_depth_wise_bn = new BatchNormalLayer(nExpansionFactor * nInputNum, stride_width);
    
    // MobileNet v2 (ReLU6)
    m_depth_wise_relu = new ReluLayer(nExpansionFactor * nInputNum * stride_width * stride_width);

    // MobileNet v2 (1x1 Conv)
    m_ConvlayerSep2 = new ConvLayer(nExpansionFactor * nInputNum , nOutputNum, stride_width, 1, 0, 1, 1); // padding: 1 , stride: 1

    // MobileNet v2 (Batch Normal)
    m_ConvSepBn2 = new BatchNormalLayer(nOutputNum, stride_width);
 

}



void layer_bottleneck::forward(float *pfInput)
{
    std::cout<<"conv1" <<std::endl;
    m_ConvlayerSep1->forward(pfInput);
    
    std::cout<<"bn1" <<std::endl;
    m_ConvSepBn1->forward(m_ConvlayerSep1->GetOutput());
    
    std::cout<<"relu1" <<std::endl;
    m_RelulayerSep1->forward(m_ConvSepBn1->GetOutput());
    
////// Identity setting  /////////////////////////////////

    std::cout<<"depth_wise 1" <<std::endl;
    m_depth_wise_conv->forward(m_RelulayerSep1->GetOutput());
    
    std::cout<<"bn-depth_wise 1" <<std::endl;
    m_depth_wise_bn->forward(m_depth_wise_conv->GetOutput());

    std::cout<<"relu-depth_wise 1" <<std::endl;
    m_depth_wise_relu->forward(m_depth_wise_bn->GetOutput());
    
    

/////Short Cut setting////////////////////////////////////
    std::cout<<"conv 2" <<std::endl;
    m_ConvlayerSep2->forward(m_depth_wise_relu->GetOutput());
    std::cout<<"bn 2" <<std::endl;
    m_ConvSepBn2->forward(m_ConvlayerSep2->GetOutput());

   // if stride is 1, we have to residual process
    if(nStride == 1){
    	for(int i =0; i< nOutputNum * nInputWidth / nStride * nInputWidth / nStride; ++i){
 		      m_ConvSepBn2->m_pfOutput[i] = m_ConvSepBn2->m_pfOutput[i] + pfInput[i];	
    	}
    
    }

}


float *layer_bottleneck::GetOutput()
{
    return m_ConvSepBn2->GetOutput();
}
