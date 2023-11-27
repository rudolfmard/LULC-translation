Multiple Map Translation
========================
This repo was forked from [MLULC](https://github.com/LBaudoux/MLULC).
The main purpose of this repository is provide the source code that was used to produce the ECOCLIMAP-SG-ML land cover map, which is used in numerical weather prediction.
Land cover maps are translated thanks to auto-encoders, as illustrated in the following figure.
ECOCLIMAP-SG-ML is obtained by map translation from ESA World Cover to ECOCLIMAP-SG+.

<img src="assets/illustration_map_translation.png" width="600" />

Installation
------------

### Software

The main dependencies of this repository are Pytorch, TorchGeo, Numpy, Pandas, h5py, netCDF4 and Matplotlib.
We recommend to use [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) with the following steps:

1. Create or clone an environment with the [Pytorch installation](https://pytorch.org/get-started/locally/) suiting your machine.
2. In this environment, install the dependencies with `pip install -r requirements.txt`
3. Clone the repository and install the package with `pip install -e .`

### Data
All data is assumed to be found in the `data` folder of the repository.
We recommend to use symbolic links to point to your storage partition.
The `data` folder should be organised as follows:

```
data
 ├── outputs        -> where the inference output will be stored
 |
 ├── saved_models   -> where the model checkpoints are stored.
 |
 ├── tiff_data      -> where the original land cover maps are stored in TIF format
 |   ├── ECOCLIMAP-SG
 |   ├── ECOCLIMAP-SG-ML
 |   ├── ECOCLIMAP-SG-plus
 |   └── ESA-WorldCover-2021
 |
 └── hdf5_data      -> where the training data is stored
     ├── ecosg.hdf5
     ├── ecosg-train.hdf5
     ├── ecosg-test.hdf5
     ├── ecosg-val.hdf5
     ├── esawc.hdf5
     └── ...
```

To download the data, a Python program will be provided.

### Check the installation

To check your installation, run the programs in the `tests` directory.



Usage
------

### Visualize maps

Once the landcovers are available in the `data/tiff_data` folder, they can be visualized using the `look_at_map.py` program.
For example, to look at ECOCLIMAP-SG-ML over the EURAT domain with a resolution of 0.1 degrees, the command is:
```
python -i look_at_map.py --lcname=EcoclimapSGML --domainname=eurat --res=0.1
```
See the header of `look_at_map.py` for more examples.

Alternatively, if you want to export maps in various formats (netCDF, DIR/HDR), the program `export_landcover.py` has a similar interface.
See the header for more information.


### Make inference

Once the landcover and the weights are correctly installed, you can perform inference on any domain for which ESA World Cover is available.
The program to make the inference is `inference_and_merging.py`.
See the documentation inside to run it.


### Reproduce results

The results presented in the manuscript can be reproduces thanks to the programs `scores_from_inference.py` and `qualitative_evaluation.py`.


### Train the model

To train the model, make sure you have set the correct parameters in a config file (a template is provide in the `config` directory).
Point to this config file in the `run.sh` program.
Then, just launch `./run.sh`.


More infos
-----------

### Repository organisation

The repository has the following directories:
  * `assets`: contains images for the documentation
  * `configs`: contains the various configuration (YAML files) for the training
  * `data`: contains all the data, as described earlier in this README
  * `drafts`: contains draft programs using the package
  * `experiments`: contains all the files created when training a model (logs, checkpoints, visualizations...)
  * `mmt`: contains the source code of the MMT package
  * `questions`: contains programs providing answers to specific questions
  * `tests`: contains programs to test the installation

Specifically, the `mmt` folder will set the organisation of the MMT package in modules and sub-modules which are as follows:
```
mmt
├── agents
│   ├── __init__.py
│   ├── base.py
│   ├── multiLULC.py
│   └── TranslatingUnet_vf.py
├── datasets
│   ├── __init__.py
│   ├── landcovers.py
│   ├── landcover_to_landcover.py
│   └── transforms.py
├── graphs
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── custom_layers
│   │   │   ├── __init__.py
│   │   │   ├── double_conv.py
│   │   │   ├── down_block.py
│   │   │   └── up_block.py
│   │   ├── attention_autoencoder.py
│   │   ├── embedding_mixer.py
│   │   ├── position_encoding.py
│   │   ├── transformer_embedding.py
│   │   ├── translating_unet.py
│   │   └── universal_embedding.py
├── inference
│   ├── __init__.py
│   ├── io.py
│   └── translators.py
└── utils
    ├── __init__.py
    ├── config.py
    ├── dirs.py
    ├── domains.py
    ├── image_type.py
    ├── misc.py
    ├── plt_utils.py
    ├── scores.py
    └── tensorboardx_utils.py
```
The modules `agents`, `graphs`, `datasets` and `utils` are mostly inherited from the MLULC repository.
The other modules are specific additions for the ECOCLIMAP-SG-ML generation.


### Class diagram

Two modules contain customised families of classes for which we provide the inheritance diagram here.
```
mmt.datasets.landcovers
├── OpenStreetMap
└── torchgeo.datasets.RasterDataset  (-> https://torchgeo.readthedocs.io/en/v0.4.1/api/datasets.html#rasterdataset)
    ├── TorchgeoLandcover
    |   ├── EcoclimapSG
    |   ├── ESAWorldCover
    |   └── EcoclimapSGplus
    |       ├── QualityFlagsECOSGplus
    |       ├── InferenceResults
    |       └── EcoclimapSGML
    |   
    └── ProbaLandcover
        └── InferenceResultsProba
```
```
mmt.inference.translators
 └── MapTranslator
     ├── EsawcToEsgp
     |    └── EsawcToEsgpProba
     ├── EsawcEcosgToEsgpRFC
     ├── MapMerger
     └── MapMergerProba
```


### Acknowledgement

Thanks to
  * [Geoffrey Bessardon](https://github.com/gbessardon) for creating the ECOCLIMAP-SG+ map and providing early releases, used as a reference in this work.
  * [Luc Baudoux](https://github.com/LBaudoux) for the initial implementation of the map translation network and the training data.
  * [Met \'Eireann](https://www.met.ie/about-us) for providing the computing facilities for this work.


### License:
This project is licensed under MIT License. See the LICENSE.txt file for details
