# Download

The synthetic dataset from the Lorenz System Generator is too heavy to be stored on GitHub. It is currently stored as part of a Dropbox folder with public access. In order to download the dataset, please run the following command from the main project directory:
```
make download-lorenz
```
This will dowload a set of timestamped folders, each containing an HDF5 file ```dataset.h5``` containing the numeric data JSON file ```metadata.json``` containing some additional information about the generative process.

In case the ```wget``` command does not work for you, please download manually the timestamped folders from [this link](https://www.dropbox.com/sh/frq47zk6ho55g4j/AAAmixPZW5elRK8pBu5TPpwPa?dl=0) and then save them into ```latentneural/data/storage/lorenz```.

## Full download
All data is available at [this link](https://www.dropbox.com/sh/frq47zk6ho55g4j/AAAmixPZW5elRK8pBu5TPpwPa?dl=0).