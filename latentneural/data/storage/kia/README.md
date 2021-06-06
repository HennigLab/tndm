# Download

The dataset with Kia Nazarpour's experiments is too heavy to be stored on GitHub. It is currently stored as part of a Dropbox folder with public access. In order to download the dataset, please run the following command from the main project directory:
```
make download-kia
```
This will dowload an HDF5 file ```dataset.h5``` containing the numeric data and the recordings from the experiments and a JSON file ```metadata.json``` containing some additional information about the experiments.

In case the ```wget``` command does not work for you, please download manually the two files from [this link](https://www.dropbox.com/sh/kapxztdx1161gco/AACVIn2CWgrw30ilRcW8Uepca?dl=0) and then save them into ```latentneural/data/storage/kia```.

## Original Data & Daily Log
All data is available at [this link](https://www.dropbox.com/sh/kapxztdx1161gco/AACVIn2CWgrw30ilRcW8Uepca?dl=0), including the original Matlab ```.mat``` files and the daily log of activities.