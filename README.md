# Fast QRS detector

**Fast QRS detector** is a Python module for detecting QRS complexes on an electrocardiogram (ECG) and distributed under the Eclipse Public License (EPL).

The development of this library started in February 2024 as part of [Aura Healthcare](https://www.aura.healthcare) project, in [OCTO Technology](https://www.octo.com/fr) R&D team.


![alt text](https://github.com/Genialmanos/QRS_Detector/blob/main/figures/Detection.png)
![alt text](https://github.com/Genialmanos/QRS_Detector/blob/main/figures/Detection2.png)


**Website** : https://www.aura.healthcare

**Github** : https://github.com/Aura-healthcare

**Version** : 0.1.2


## Installation / Prerequisites

#### User installation

The easiest way to install fast QRS detector is using ``pip`` :

    $ pip install fast_qrs_detector

#### Dependencies

**Fast QRS detector** requires the following:
- Python (>= 3.6)
- numpy >= 1.16.0
- scipy >= 1.2.0
- matplotlib >= 3.3.4

Note: The package can be used with all Python versions from 3.6 to latest version (currently Python 3.11).


## Getting started

### Fast QRS detector

This package provides methods to detect QRS complexes on an electrocardiogram (ECG) like MIT-BIH-Arrhythmia.

```python
from fast_qrs_detector import qrs_detector

qrs = qrs_detector(signal, frequence_sampling)
```

### Plot functions

```python
from fast_qrs_detector import print_signal_with_qrs
# Basic usage
print_signal_with_qrs(signal, qrs_predicted)

# Full usage with all parameters
print_signal_with_qrs(signal, qrs_predicted, true_qrs=labeled_qrs, mini=10000, maxi=15000, description="QRS between frames 10000 and 15000")
```
Parameters:

- signal (list or numpy array): The 1D array of signal values (e.g., ECG signal data).
- qrs_predicted (list of integers): Indices of the signal where QRS complexes are predicted.
- true_qrs (list of integers, optional): Indices of the signal where QRS complexes are actually located (true labels). Default is an empty list.
- mini (int, optional): The starting index of the segment of the signal to be plotted. Default is 0.
- maxi (int, optional): The ending index of the segment of the signal to be plotted. If set to 1, the function plots up to the end of the signal. Default is 1.
- description (str, optional): A description or title for the plot. Default is an empty string.

## Quality control and performances

[This algorithm uses a benchmark previously made by Aura.](https://github.com/ecg-tools/benchmark-qrs-detectors)
 It compares different public libraries in terms of accuracy on 5 public datasets. This algorithm achieves better results in terms of accuracy, as well as efficiency. Compared with the best of the algorithms tested, this algorithm achieves better results with an average time 5 times shorter. [More details here.](README2.md)


## References

Here are the main references used to made this algorithm:

> Zidelmal, Z., Amirou, A., Adnane, M., & Belouchrani, A. (2012). QRS detection based on wavelet coefficients. Computer Methods And Programs In Biomedicine, 107(3), 490‑496. https://doi.org/10.1016/j.cmpb.2011.12.004

> Lu, X., Pan, M., & Yu, Y. (2018). QRS detection based on improved adaptive threshold. Journal Of Healthcare Engineering, 2018, 1‑8. https://doi.org/10.1155/2018/5694595 

> Modak, S., Abdel-Raheem, E., & Taha, L. Y. (2021). A novel adaptive multilevel thresholding based algorithm for QRS detection. Biomedical Engineering Advances, 2, 100016. https://doi.org/10.1016/j.bea.2021.100016 

> M. Šarlija, F. Jurišić and S. Popović (2017). "A convolutional neural network based approach to QRS detection," Proceedings of the 10th International Symposium on Image and Signal Processing and Analysis,121-125 https://doi.org/10.1109/ISPA.2017.8073581 

## Author

**Jean-Charles Fournier** - (https://github.com/Genialmanos)


## License

This project is licensed under the *Eclipse Public License - v 2.0* - see the [LICENSE.md](https://github.com/Genialmanos/QRS_Detector/blob/main/LICENSE) file for details

## Acknowledgments

I hereby thank Clément Le Couedic and Fabien Peigné, my coworkers who gave me time to Open Source this library.
