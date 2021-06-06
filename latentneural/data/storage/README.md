# Download

The synthetic dataset from the Lorenz System Generator and the dataset with Kia Nazarpour's experiments are too heavy to be stored on GitHub. They are currently stored as part of a Dropbox folder with public access. In order to download the datasets, please run the following commands from the main project directory:
```
make download-lorenz
make download-kia
```
This will dowload all datesets as ```dataset.h5``` files and their metadata as ```metadata.json``` files.

In case the ```wget``` command does not work for you, please download manually the timestamped folders from [this link](https://www.dropbox.com/sh/h9h0o1rllx8ggus/AAAyo4uaoEcRmnMB3UZeK9CGa?dl=0) and then save them into ```latentneural/data/storage```.