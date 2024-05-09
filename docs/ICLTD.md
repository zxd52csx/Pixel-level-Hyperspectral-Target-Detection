# Implicit Contrastive Learning-based Target Detector (ICLTD)

[Table of Content of ICLTD](./ICLTD.png)

Although optimization methods based on pseudo data have been widely used by researchers in the field, under the influence of spectral variability, ensuring the authenticity of pseudo data and the accuracy of labels is challenging. Inaccurate features can lead to model training biases and degrade detection performance. Additionally, generating pseudo data also consumes resources, making the detection process complex. 

To address above challenges, we propose an optimization method based on implicit contrastive learning and introduce a HTD model based on implicit contrastive learning. The proposed ICLTD does not require any pseudo data; it only needs a small amount of prior spectra and unlabeled real spectral from test HSIs for training and inference. To improve target detectability, we design a local spectral similarity constraint (LSSC).

## Get started

Train and evaluate ICLTD with LSSC (WSD on ABU-datasets) :

```python
python ICLTD.py --LSSC_weight 1 --ICL_weight 1 --weight_decay 0 #default 5e-4
```

Train and evaluate ICLTD with LSSC (CSD on MT-ABU-datasets) :

```python
python ICLTD.py --cross_scene --LSSC_weight 1 --ICL_weight 0.5 
```

Improve the intensity of implicit contrastive learning would improve the background suppression capability of the detector. On the contrary, reduce the intensity would improve the target detectability. You can set different value of 'ICL_weight' to adjust the intensity of implicit contrastive learning:

We designed two kinds of feature extraction networks (fully connected-based networks & self attention-based networks), you can choose as follows:

```python
python ICLTD.py --cross_scene --LSSC_weight 1 --ICL_weight 0.5 --model fc
python ICLTD.py --cross_scene --LSSC_weight 1 --ICL_weight 0.5 --model trans
```

other parameters:

```python
--eo # compare with other methods
--LSSC_t # candidate threshold of LSSC default 0.1
```

