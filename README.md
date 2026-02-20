# P-PFDSTL

This repository contains the official implementation of the paper "Privacy-Preserving Federated Dynamic Spatio-Temporal Learning for Traffic Flow Prediction in Consumer IoT Networks".

## 📊 Datasets

<div align="center">

| Attributes   | PeMS03         | PeMS04         | PeMS07         | PeMS08         |
|--------------|----------------|----------------|----------------|----------------|
| Time Spans   | 9/1-11/30/2018 | 1/1-2/28/2018  | 5/1-8/31/2017  | 7/1-8/31/2016  |
| Sensors      | 358            | 307            | 883            | 170            |
| Intervals    | 5 min          | 5 min          | 5 min          | 5 min          |
| Road Types   | Highway        | Highway        | Highway        | Highway        |
| Regions      | California     | California     | California     | California     |

</div>

## 📂 Directory Structure

```plaintext
📂 P-PFDSTL/
│
├── 📝 embedding.py           # Traffic embedding for spatial-temporal data
├── 📝 DSGL.py                # Spatio-temporal feature extraction
├── 📝 FL.py                  # Federated learning client and aggregation logic
└── 📝 main1.py               # Main entry for training and evaluation
```

## 📦 Requirements

- torch
- datetime
- numpy
- pandas
- math
- scipy
- torchvision

## 🚀 Running Experiments

 ```bash
     python main1.py --root <path_to_dataset> --dataset <dataset_name> --device <device> --seed <random_seed>
 ```
