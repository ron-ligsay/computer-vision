# Semantic Segmentation of ariel (satellite) imagery using U-net
semantic segmentation is a computer vision task where we classify each pixel in an image into a class. In this project, we will use a U-Net architecture to segment satellite images of Dubai, the UAE into 6 classes. The classes are: Buildings, Land, Road, Vegetation, Water, and Unlabeled.

### Color Hexes
Building: #3C1098
Land (unpaved area): #8429F6
Road: #6EC1E4
Vegetation: #FEDD3A
Water: #E2A929
Unlabeled: #9B9B9B

## Preprocess
* Images come in many sizes: 797x644, 509x544, 682x658, 1099x846, 1126x1058, 859x838, 1817x2061,  2149x1479​

* Need to preprocess so we can capture all images in a single batch.
    => cropt to a size divisible by 256 and extract patches of size 256x256

* Masks are RGB and information provided as HEX color code.
    => need to convert hex to rgb values and then convert rgb labels to integer values and then to one hot encoded

## Predict
predicted (segmented) images need to converted back into original rgb colors

## References and Dataset
from DigitalSreeni video: https://www.youtube.com/watch?v=jvZm8REF2KY
nsreenu repo: https://github.com/bnsreenu/python_for_microscopists
dataset from: https://www.kaggle.com/tarunpaparaju/dubai-uae-semantic-segmentation