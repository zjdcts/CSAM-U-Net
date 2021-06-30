# CSAM-U-Net
This is the source code of paper 《CSAM U-Net: Multimodal Brain Tumor Segmentation Using Channel-Spatial Attention Module》. 
This code is based on https://github.com/ellisdg/3DUnetCNN



Figure 1: Overall architecture of the proposed CSAM U-Net.

![csam_unet](D:\PycharmProjects\CSAM-U-Net\figs\csam_unet.png)



Figure 2: Overall architecture of the proposed CSAM U-Net.

![CSAM](D:\PycharmProjects\CSAM-U-Net\figs\CSAM.png)



Figure 3: Diagram of each attention sub-module.As illustrated, the channel sub-module first concatenates max-pooling outputsand average-pooling outputs, then forward to a transformer layer, convolution layer and activation layer; the spatial sub-moduleconcatenates similar two outputs that are pooled along the channel axis and forward them to a downsample layer, upsample layerand activation layer.

![CSAM_detail](D:\PycharmProjects\CSAM-U-Net\figs\CSAM_detail.png)



Figure 5: Boxplot of the overall $Dice$ performance on ET, WT and TC for different methods.

![DiceCompare](D:\PycharmProjects\CSAM-U-Net\figs\DiceCompare.png)



Figure 6: Boxplot of the overall $Sensitivity$ performance on ET, WT and TC for different methods.

![SensitivityCompare](D:\PycharmProjects\CSAM-U-Net\figs\SensitivityCompare.png)



Figure 7: Segmentation results of the peritumoral edema (ED, green region), GD-enhancing tumor (ET, blue region) and necroticand non-enhancing tumor core (NCR/NET, red region) structures for different methods. (a) Gound Truth; (b) CSAM U-Net(ours); (c) 3D U-Net; (d) Cascade U-Net; (e) Attention U-Net.

![SensitivityCompare](D:\PycharmProjects\CSAM-U-Net\figs\ResultCompare.png)


