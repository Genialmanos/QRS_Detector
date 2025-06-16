# Research & Development Documentation

## 1. Introduction

This document details the research and development process undertaken to select the optimal QRS detection algorithm for our Python library. A comparative study was conducted, evaluating various popular algorithms from Python libraries and research papers based on their performance across diverse ECG datasets. The goal was to identify an algorithm that balances high accuracy with computational efficiency, suitable for real-world applications.

## 2. Methodology

### 2.1 Evaluation Approach

Our methodology involved implementing and testing several QRS detection algorithms identified in a preliminary study conducted by [Aura in 2022](#ref1). The performance of each algorithm was rigorously evaluated using multiple publicly available ECG datasets, ensuring the reproducibility of our results. Performance was primarily measured using the F1-score, calculated with a tolerance of 100ms around the R-peak annotations provided by medical experts. For multi-channel ECG recordings, only the most relevant channel (which could vary) was used for performance evaluation.

### 2.2 Datasets

To ensure robustness and generalizability, we selected five open-data ECG datasets with varying characteristics, including different sampling frequencies, signal lengths, noise levels, and underlying pathologies:

1.  **MIT-BIH Arrhythmia Database (f=360 Hz):** A widely used benchmark dataset featuring 48 half-hour recordings with significant signal variations, noise, and various arrhythmias.
2.  **MIT-BIH Supraventricular Arrhythmia Database (f=128 Hz):** Contains 78 half-hour recordings focusing on supraventricular arrhythmias, which can alter QRS complex morphology.
3.  **MIT-BIH Noise Stress Test Database (f=360 Hz):** Includes two ECG recordings with artificially added noise at various signal-to-noise ratios (levels n\_00 to n\_24, where n\_00 is the noisiest). This helps assess algorithm performance under extreme noise conditions.
4.  **MIT-BIH Long-Term ECG Database (f=128 Hz):** Comprises 7 recordings, each lasting 14 to 22 hours, testing algorithm efficiency on extended signals.
5.  **European ST-T Database (f=250 Hz):** Contains 89 two-hour recordings from patients with various suspected cardiac conditions. Its sampling frequency is relevant to specific sensor applications (like Aura's), and it tests robustness against unusual ECG patterns.

More details on the source of the datasets and their use in our tests can be found here: https://github.com/ecg-tools/benchmark-qrs-detectors

### 2.3 Initial Benchmark: Hamilton Algorithm

Initial tests by Aura identified the **Hamilton algorithm** (implemented in the `scipy` library) as providing excellent [accuracy](#ref1). This algorithm, developed shortly after the seminal Pan-Tompkins algorithm, involves filtering, peak detection, event vector creation (signal level, time since last peak), QRS/noise annotation based on adaptive thresholds, and a 200ms refractory period. However, its computational time was noted as a potential constraint for real-time or resource-constrained applications.

## 3. Algorithms Evaluated

To find an alternative potentially offering better performance or efficiency, we implemented and tested the following algorithms:

* **Convolutional Neural Network (CNN):** Based on [Šarlija et al. (2017)](#ref4).
* **Dynamic Threshold:** Based on [Lu et al. (2018)](#ref5).
* **Hamilton:** As implemented in `scipy`, based on Hamilton & Tompkins (derived from [Pan & Tompkins, 1985](#ref6)).
* **Multi-level Thresholding (Multi-lvl):** Based on [Modak et al. (2021)](#ref7).
* **Wavelet Transform:** Based on [Zidelmal et al. (2012)](#ref8).
* **Fuzzy Clustering:** Based on [Pander (2022)](#ref9).
* **Empirical Mode Decomposition (EMD):** Based on [Slimane & Naït-Ali (2010)](#ref10).
* **Low Computation Method (Low comput):** Based on [Yakut & Bolat (2018)](#ref11).

## 4. Experimental Results

### 4.1 Accuracy (F1-Score)

The initial F1-scores (with 100ms tolerance) for each algorithm across the datasets were as follows:

| Algo              | Arrhythmia | Long term |  n_00 |  n_06 |  n_12 |  n_18 |  n_24 | Supraventricular | European-st | F1 mean |
|:-----------------:|:---------: |:---------:|:-----:|:-----:|:-----:|:-----:|:-----:| :--------------: |:-----------:|:-------:|
| CNN               |      98.71 |         / | 87.92 | 95.53 | 98.75 | 99.57 | 99.77 |            99.32 |       98.91 |   98.312 |
| Dynamic Threshold |      98.97 |     99.23 | 77.00 | 90.65 | 98.75 | 99.94 | 99.95 |            99.60 |       99.17 |**98.046**|
| Hamilton          |      98.94 |     99.45 | 79.80 | 92.94 | 99.17 | 99.96 | 99.99 |            98.85 |       97.71 |  97.864 |
| Multi-lvl         |      98.76 |     68.50 | 81.48 | 90.59 | 98.27 | 99.88 | 99.87 |            97.80 |       97.40 |  91.296 |
| Wavelet           |      98.14 |     92.96 | 80.07 | 89.73 | 97.05 | 99.79 | 99.94 |            94.30 |       96.50 |  95.043 |
| Clustering        |      98.84 |     92.27 | 79.05 | 89.47 | 95.19 | 97.59 | 98.28 |            96.00 |       96.95 |  95.195 |
| EMD               |      94.31 |     87.00 | 75.52 | 87.94 | 95.06 | 96.62 | 97.01 |            88.19 |       94.84 |  90.954 |
| Low comput        |      98.20 |     94.94 | 74.86 | 82.84 | 86.91 | 89.09 | 89.85 |            94.24 |       92.20 |  92.858 |

**Observations:**

* Performance often differed from results reported in the original publications. This could be due to implementation challenges stemming from ambiguous parameter descriptions, lack of rigorous or standardized testing protocols in the papers, and the absence of open-source implementations for validation.
* Algorithm performance varied significantly across datasets, particularly concerning different sampling frequencies (many algorithms were originally designed for 360 Hz) and noise levels. The **CNN** showed robustness to noise but struggled with the long-term dataset in this initial run. The **Dynamic Threshold** and **Hamilton** algorithms showed strong overall performance.

### 4.2 Execution Time

Execution times were measured on a machine with an 11th Gen Intel® Core™ i7-1185G7 × 8 processor, 32 GiB RAM, and Mesa Intel® Xe Graphics. The total execution time (in seconds) across datasets is shown below.
**Note:** These timings reflect the core algorithm processing on raw numerical signal data (similar to NumPy array input). Additional overhead may be introduced when using Pandas DataFrames, especially with NaN interpolation and segmentation features activated.

`time / long` excludes the time-consuming 'long term' dataset for a focused comparison.

| Algo              | Arrhythmia | Long term | n_00 | n_06 | n_12 | n_18 | n_24 | Supraventricular | European-st | time total | time / long |
|:-----------------:|:---------: |:---------:|:----:|:----:|:----:|:----:|:----:| :--------------: |:-----------:|:----------:|:-----------:|
| Multi-lvl         |         40 |      2400 |    6 |    6 |    6 |    6 |    6 |               48 |         585 |       3103 |         703 |
| Low comput        |         50 |      3900 |    6 |    6 |    6 |    6 |    6 |               42 |         600 |       4622 |         722 |
| Wavelet           |         48 |      3600 |    7 |    4 |    4 |    7 |    4 |               50 |         700 |       4424 |         824 |
| Dynamic Threshold |         53 |      5520 |    6 |    6 |    6 |    6 |    6 |               53 |         700 |   **6356** |     **836** |
| Hamilton          |        171 |      5000 |   10 |   15 |   15 |   10 |   10 |              240 |        2700 |       8171 |        3171 |
| EMD               |        490 |      6900 |   30 |   30 |   30 |   30 |   30 |              250 |        5760 |      13550 |        6650 |
| Clustering        |        720 |      9000 |   30 |   30 |   30 |   30 |   30 |             1980 |        3360 |      15210 |        6210 |
| CNN               |      10800 |    151800 |  660 |  660 |  660 |  660 |  660 |             5500 |      126720 |     298120 |      146320 |

**Observations:**

* There is a vast difference in computational cost between algorithms.
* **CNN**, while accurate, is computationally prohibitive for many applications.
* Algorithms like **EMD** and **Clustering** are also significantly slower than threshold-based or wavelet methods.
* The **Dynamic Threshold** algorithm offers comparable accuracy to **Hamilton** but runs approximately 4 times faster (comparing `time / long`).

## 5. Analysis and Algorithm Selection

Based on the combined accuracy and execution time results:

* **CNN, EMD, Clustering:** Excluded due to excessive execution time.
* **Low Comput:** Excluded due to relatively lower F1 scores compared to top performers.
* **Hamilton:** While accurate, its execution time is significantly higher than the Dynamic Threshold algorithm for similar F1 scores.
* **Multi-lvl, Wavelet, Dynamic Threshold:** Identified as the most promising candidates balancing accuracy and speed, warranting further investigation and optimization.

### 5.1 Optimized Results

Further tuning of parameters was performed on the selected candidates. The "Dynamic Threshold +" represents an optimized version of the Dynamic Threshold algorithm (refinement of some of the parameters given in the article). 

| Algo                    | Arrhythmia | Long term |  n_00 |  n_06 |  n_12 |  n_18 |  n_24 | Supraventricular | European-st | F1 mean   |
|:-----------------------:|:---------: |:---------:|:-----:|:-----:|:-----:|:-----:|:-----:| :--------------: |:-----------:|:---------:|
| **Dynamic Threshold +**|  **99.11** | 99.22 | 79.40 | 91.68 | 98.85 | 99.92 | 99.95 |      **99.50**   |   99.07 |**98.172** |
| Dynamic Threshold       |      98.97 |     99.23 | 77.00 | 90.65 | 98.75 | 99.94 | 99.95 |            99.60 |       **99.17** |  98.046   |
| Hamilton                |      98.94 |     **99.45** | **79.80** | **92.94** | **99.17** | **99.96** | **99.99** |            98.85 |       97.71 |  97.864   |

**Final Analysis:** The optimized **Dynamic Threshold algorithm** demonstrated the best overall F1-score performance across the diverse datasets while maintaining significantly better computational efficiency compared to the original Hamilton benchmark.

## 6. Conclusion

Based on this comparative analysis evaluating both accuracy (F1-score) and execution speed across multiple challenging ECG datasets, the **Dynamic Threshold +** was selected for implementation in this library. It provides state-of-the-art detection accuracy comparable to or exceeding other methods, including the widely used Hamilton algorithm, but with substantially lower computational requirements, making it suitable for a broader range of applications.

## 7. References
<a id="ref1"></a>1.  Benchmark QRS Detectors Repository: <https://github.com/ecg-tools/benchmark-qrs-detectors> (Referenced for Aura's 2022 study and for Hamilton algorithm accuracy in initial tests)

<a id="ref4"></a>
2.  Šarlija, M., Jurišić, F., & Popović, S. (2017). A convolutional neural network based approach to QRS detection. *Proceedings of the 10th International Symposium on Image and Signal Processing and Analysis*, 121–125. <https://doi.org/10.1109/ISPA.2017.8073581>

<a id="ref5"></a>
3.  Lu, X., Pan, M., & Yu, Y. (2018). QRS detection based on improved adaptive threshold. *Journal Of Healthcare Engineering*, 2018, 1–8. <https://doi.org/10.1155/2018/5694595>

<a id="ref6"></a>
4.  Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS detection algorithm. *IEEE Transactions On Bio-medical Engineering*, BME-32(3), 230–236. <https://doi.org/10.1109/tbme.1985.325532>

<a id="ref7"></a>
5.  Modak, S., Abdel-Raheem, E., & Taha, L. Y. (2021). A novel adaptive multilevel thresholding based algorithm for QRS detection. *Biomedical Engineering Advances*, 2, 100016. <https://doi.org/10.1016/j.bea.2021.100016>

<a id="ref8"></a>
6.  Zidelmal, Z., Amirou, A., Adnane, M., & Belouchrani, A. (2012). QRS detection based on wavelet coefficients. *Computer Methods And Programs In Biomedicine*, 107(3), 490–496. <https://doi.org/10.1016/j.cmpb.2011.12.004>

<a id="ref9"></a>
7.  Pander, T. (2022). A new approach to adaptive threshold based method for QRS detection with fuzzy clustering. *Biocybernetics And Biomedical Engineering*, 42(1), 404–425. <https://doi.org/10.1016/j.bbe.2022.02.007>

<a id="ref10"></a>
8. Slimane, Z. H., & Naït-Ali, A. (2010). QRS complex detection using Empirical Mode Decomposition. *Digital Signal Processing*, 20(4), 1221–1228. <https://doi.org/10.1016/j.dsp.2009.10.017>

<a id="ref11"></a>
9. Yakut, Ö., & Bolat, E. D. (2018). An improved QRS complex detection method having low computational load. *Biomedical Signal Processing And Control*, 42, 230–241. <https://doi.org/10.1016/j.bspc.2018.02.004>
