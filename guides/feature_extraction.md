# Feature Extraction Process

## LSMDC

To begin with, request access to the [MPII Movie Description dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset/request-access-to-mpii-movie-description-dataset/) and download the dataset.
Then, we sample each frame in 24fps and resize it to 224x224 so that the input shape is compatible with I3D and ResNet models.
Finally, we use two types of visual features for LSMDC:

* [I3D](https://github.com/piergiaj/pytorch-i3d): We provide the whole video as an input to obtain the output of the AvgPool layer. Note that we use Kinetics-pretrained model weight to extract I3D features.
* ResNet-101: We sample each frame in 3fps for temporal consistency with I3D (i.e., a factor of 8) and use pretrained weight from the PyTorch model hub to obtain the output of the AvgPool layer.


## VIST

Download [the dataset](http://visionandlanguage.net/VIST/dataset.html) first. We use three types of visual features for VIST:

* ResNet-101: As in the LSMDC dataset, we use pretrained weight from the PyTorch model hub to obtain the output of AvgPool layer.
* [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch): We use VisualGenome-pretrained model weight to extract at most 20 object features with the highest likelihood per image. If less than 20 objects are detected in an image, we zero-pad the rest for consistency in dimension.
* [ViLBERT](https://github.com/jiasenlu/vilbert_beta): Using the R-CNN feature mentioned above as visual input and empty string as language input, we extract the output of the visual pooling layer.

