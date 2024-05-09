# Self-supervised Deep Clustering-based Target Detector (SSDCTD) 

[Table of content of SSDCTD](./SSDCTD.png)

The implicit contrastive learning-based hyerspectral target detector is optimized without negative samples. It leverages the inherent feature differences between real spectral (backgrounds and targets) and prior spectra to guide the model in learning distinctive spectral feature representations, thereby achieving differentiation between target and background spectra. However, due to spectral variability, the feature differences between prior spectra and target spectra can lead to a decrease in target detectability.

To enhance target detection performance under spectral variability conditions, we designed a spectral variability mining method based on self-supervised learning and a proxy task based on deep clustering. This method leverages the inherent feature differences between target and background spectra. By utilizing spectral reconstruction loss and clustering loss functions, it clusters target and background spectra into different categories, thus improving the detection confidence of target samples with spectral variations and enhancing the model's target detection performance.

## Get started

Train and evaluate SSDCTD with subset division (WSD on ABU-datasets) :

```python
python SSDCTD.py --SD 
```

If you want to enhance the background suppression, enlarge the intensity of implicit contrastive learning as follows:

```python
python SSDCTD.py --SD --ICL_weight 1 # default 0.5
```

Subset division is realized under the help of other HTD detectors (such as DS-$SA^2$). However, these detectors do not perform well in the cross-scene detection way. Therefore, it is recommended to skip the subset division process in the CSD way:

```python
python SSDCTD.py --cross_scene --weight_decay 0 # default 5e-4
```

You could add the local spectral similarity constraint to the SSDCTD:

```python
python SSDCTD.py --cross_scene --LSSC_weight 1 --weight_decay 0
```

other parameters:

```python
--window_size 10 # window size for subset division
--eo # compare with other methods
--LSSC_t # candidate threshold of LSSC default 0.1
--model trans # choose model (default fc)
```

