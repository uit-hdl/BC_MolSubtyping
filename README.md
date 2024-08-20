# BC_MolSubtyping
This is a modified repository of MultiStainDeepLearning by Foersch et al.

This is the GitHub Repository for the classification of breast H&E-stained histopathology images. It was adapted from https://github.com/AGFoersch/MultiStainDeepLearning and modified as needed. This repository can be used for any image classification task.


Our paper: [Deep learning-based classification of breast cancer molecular subtypes from H&E whole-slide images](https://www.nature.com/articles/s41591-022-02134-1)

Original paper: [Multistain deep learning for prediction of prognosis and therapy response in colorectal cancer](https://www.nature.com/articles/s41591-022-02134-1)


 
## Getting started

### Dependencies
This project requires Python 3 (3.6.9) with the following additional packages:
* [PyTorch](https://pytorch.org/) (torch==1.9.0, torchvision==0.10.0) with CUDA support
* [NumPy](https://numpy.org/) (1.18.1)
* [tqdm](https://github.com/tqdm/tqdm) (4.64.0)
* [lifelines](https://github.com/CamDavidsonPilon/lifelines) (0.25.7)
* [scikit-learn](https://scikit-learn.org/stable/) (0.24.2)
* [matplotlib](https://matplotlib.org/) (3.3.4)
* [pandas](https://pandas.pydata.org/) (0.24.2)
* [TensorBoard](https://pypi.org/project/tensorboard/) (2.1.0)
* [Pillow](https://pypi.org/project/Pillow/) (7.1.2)
* [Captum](https://captum.ai/) (0.4.0)


### General usage
For getting started and general usage of the codes, you can check the original repository here: https://github.com/AGFoersch/MultiStainDeepLearning.

To use your own data for the classification task, you will need to create your own .csv files describing the data and .json files describing the configuration as shown below. 

#### Data description (.csv)
Each modality requires at least two .csv description files (one for training, one for validation) structured like so:

| Patient_ID | Path                 | Label | Set   |
|------------|----------------------|-------|-------|
| pid0       | path/to/pid0_0.jpg   | A     | TRAIN |
| pid0       | path/to/pid0_1.jpg   | A     | TRAIN |
| ...        | ...                  | ...   | ...   |
| pid999     | path/to/pid999_0.jpg | B     | TRAIN |
| pid999     | path/to/pid999_1.jpg | B     | TRAIN |

Some notes:
* All entries using the same Patient_ID must share the same label.
* The .csv file for the training data needs each entry to have the value `TRAIN` in the `Set` column. The files for validation or testing need to have the value `VALID` in the `Set` column.
* The `Patient_ID`, `Path` and `Set` column names must exist exactly with this spelling. The `Label` column, on the other hand, can be named however you like and the labels themselves can have whatever names you like (just be consistent, of course).
* If you specify the `data_root` value in your configuration files, the paths in the `Path` column will be treated as relative to that.

#### Configuration (.json)
The configuration .json files that describe model training and evaluation are structured as follows (minus the comments):
```json
{
    "name": "My_setup",                                                            // Experiment name, will be appended to save_dir (see trainer).
    "n_gpu": 1,                                                                                 // Number of GPUs, should stay at 1.

    "arch": {
        "type": "MultiModel",                                                                   // Model architecture class name. Both uni- and multimodal training uses this class.
        "args": {
          "num_classes": 2,                                                                     // Number of classes, should be >= 2.
          "lo_dims": [32,32,32],                                                                // Number of output features for each unimodal model.
                                                                                                // --> Only one entry for unimodal training.
          "lo_pretrained": [                                                                    // Paths to the pretrained unimodal models. Omit this entry for unimodal training.
            "./saved/path/to/checkpoint/for/modality0.pth"
          ],
          "mmhid": 64,                                                                          // Input size for the final classification layer.
          "dropout_rate": 0.33                                                                  // Dropout rate only applies during multimodal training.
        }
    },
    "data_loader": {
        "type": "BasicMixDataLoader",                                                           // Data loader class name.
        "args":{
          "data_root": ".data/",                                                                // Prefix for all paths in the data description .csv files.
          "dataframes": [                                                                       // Paths to .csv files describing the training data, one for each modality in use.
            "./config/csv/training_dataframe_for_modality0.csv"
          ],
          "labels": ["Tumor_Label"],                                                            // Name of the label column in your .csv files
          "dataframe_valid": "./config/csv/multimodal_val.csv",                                 // Path to data description used for validation/early stopping.
                                                                                                // These are different files for the unimodal and multimodal cases, see the previous section.
          "valid_columns": ["modality0_Path"],                                                  // Name of the columns containing the paths for each modality.
                                                                                                // In the unimodal case, the list contains only "Path".
          "shuffle": true,                                                                      // data shuffling only applies to training data.
          "num_workers": 8,
          "batch_size": 128                                                                      // One dataset element contains one image from every modality in use
                                                                                                // --> Batches take up more memory during multimodal training.
       }
  },
    "transformations":{
      "type": "MyImageAugmentation",                                                              // See datahandler/transforms/data_transforms.py for possible options.
      "args": {
        "size": 512
      }
    },
    "optimizer": {                                                                              // Accepts any optimizer from torch.optim along with its arguments.
        "type": "Adam",
        "args":{
            "lr": 0.00001
        }
    },
    "loss": {                                                                                   
        "type": "CrossEntropyLoss",
        "args": {
        }
    },

    "metrics": {                                                                                // Metrics to track during training.
      "epoch": [
        {
          "type":  "accuracy_epoch",
          "args":  {}
      }
    ],
      "running": []
    },
    "trainer": {
        "type": "BasicMultiTrainer",
        "args": {},
        "epochs": 1000,                                                                         // Number of epochs to train for.
        "save_dir": "saved/",                                                                   // Prefix for the save directory (see "name" key)
        "save_period": 10,                                                                      // Save a model checkpoint every x epochs.
        "val_period": 1,                                                                        // Validate model every x epochs, save a checkpoint if best performance yet.
        "verbosity": 2,                                                                         // Between 0 and 2. 0 is least verbose, 2 most.
        "freeze": true,                                                                         // Freeze unimodal models during multimodal training?
        "unfreeze_after": 50,                                                                   // If freeze is true, unfreeze unimodal models after this many epochs.
        "monitor": ["max val_accuracy_epoch", "min val_loss_epoch"],                            // Metrics to monitor for determining best performance.
        "tensorboard": true,                                                                    // Track training with TensorBoard?
        "evaluation": true                                                                      // Evaluate model performance on validation data when finished with training?
    }
}
```


## Citation
If any part of this code is used, please give appropriate citation to original authors paper:
Foersch, S., Glasner, C., Woerl, AC. et al. Multistain deep learning for prediction of prognosis and therapy response in colorectal cancer. Nat Med (2023). https://doi.org/10.1038/s41591-022-02134-1


## License
This project is licensed under the GNU GPLv3 license.


## Acknowledgements
This project's structure is based on the [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by Victor Huang, licensed under the MIT License - see LICENSE-3RD-PARTY for details."#BC_MolSubtyping#" 
