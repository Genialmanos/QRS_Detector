# QRS_Detector

**QRS_Detector** is a Python module for detecting QRS complexes on an electrocardiogram (ECG) and distributed under the Eclipse Public License (EPL).

The development of this library started in February 2024 as part of [Aura Healthcare](https://www.aura.healthcare) project, in [OCTO Technology](https://www.octo.com/fr) R&D team.


![alt text](https://github.com/Genialmanos/QRS_Detector/blob/main/figures/Detection.png)
![alt text](https://github.com/Genialmanos/QRS_Detector/blob/main/figures/Detection2.png)


**Website** : https://www.aura.healthcare

**Github** : https://github.com/Aura-healthcare

**Version** : 1.0.0


## Installation / Prerequisites

#### User installation

The easiest way to install QRS_Detector is using ``pip`` :

    $ pip install QRS_Detector

you can also clone the repository:

    $ git clone https://github.com/Genialmanos/QRS_Detector.git
    $ python setup.py install

#### Dependencies

**QRS_Detector** requires the following:
- Python (>= 3.6)
- numpy >= 1.16.0
- scipy >= 1.2.0

Note: The package can be used with all Python versions from 3.6 to latest version (currently Python 3.11).


## Getting started

### QRS detection

This package provides methods to detect QRS complexes on an electrocardiogram (ECG) like MIT-BIH-Arrhythmia.

```python
from QRS_Detector import qrs_detector

qrs = qrs_detector(signal, frequence_sampling)
```

### Plot functions

Ajout d'une fonction pour plot ?

## Quality control and performances

vs hamilton avec github de marysa
renvoie vers un autre readme + détaillé


## References

Here are the main references used to made this algorithm:

> Zidelmal, Z., Amirou, A., Adnane, M., & Belouchrani, A. (2012). QRS detection based on wavelet coefficients. Computer Methods And Programs In Biomedicine, 107(3), 490‑496. https://doi.org/10.1016/j.cmpb.2011.12.004

> Lu, X., Pan, M., & Yu, Y. (2018). QRS detection based on improved adaptive threshold. Journal Of Healthcare Engineering, 2018, 1‑8. https://doi.org/10.1155/2018/5694595 

> Modak, S., Abdel-Raheem, E., & Taha, L. Y. (2021). A novel adaptive multilevel thresholding based algorithm for QRS detection. Biomedical Engineering Advances, 2, 100016. https://doi.org/10.1016/j.bea.2021.100016 

> M. Šarlija, F. Jurišić and S. Popović (2017). "A convolutional neural network based approach to QRS detection," Proceedings of the 10th International Symposium on Image and Signal Processing and Analysis,121-125 https://doi.org/10.1109/ISPA.2017.8073581 

## Authors

**Jean-Charles Fournier** - (https://github.com/Genialmanos)


## License

This project is licensed under the *Eclipse Public License - v 2.0* - see the [LICENSE.md](https://github.com/Genialmanos/QRS_Detector/blob/main/LICENSE) file for details

## How to contribute
Please refer to [How To Contribute](https://github.com/Aura-healthcare/hrv-analysis/blob/master/CONTRIBUTING.md)

## Acknowledgments

I hereby thank Clément Le Couedic and Fabien Peigné, my coworkers who gave me time to Open Source this library.
