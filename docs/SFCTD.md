# Siamese Fully Connected-based Target Detector (SFCTD) 

[Table of content of SFCTD](./SFCTD.png)

The motivation behind SFCTD is to build a neural network-based target detector in situations where training samples are limited. To enable the model to be trained, we devised a pseudo data generation method. The model is trained through supervised contrastive learning, establishing contrastive learning between the generated pseudo targets/background samples and the prior spectra. Since different initialization parameters can affect the model's performance, we introduced model ensemble techniques to enhance detection performance. 

## Get started

Train and evaluate SFCTD in the WSD way on ABU-datasets:

```python
python SFCTD.py --batchsize 32 --epoch 20
```

Train and evaluate SFCTD in the CSD way on MT-ABU-datasets:

```python
python SFCTD.py --cross_scene --batchsize 256 --epoch 10
```

Compared with other classical methods:

```python
python SFCTD.py --batchsize 32 --eo --epoch 20
```

other parameters:

```python
--ensemble_num 4 # more ensemble number may improve the detection performance
```

Because the data range of detection result predicted by SFCTD is from 0-1, it is not necessary to normalize the detection results. However, normalization would improve the separability of targets and backgrounds.  
