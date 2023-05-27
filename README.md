# Set Up:

We assume you have a working installation of conda (If not, see the docs at https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html for installation instructions -- we recommend using the miniconda installer), and an NVIDIA GPU with drivers installed.

The commands below are geared for a Unix environment. If you are on Windows, you will need to change the path to the bootstrap env in the first line of the script, and remove the bootstrap environment directory at the end of the script without using the `rm` command.

Copy and paste the following commands into a terminal window to create a conda environment with all of the required dependencies:
```
# clone the repo
git clone ...

# cd into the repo
cd poincare_deep_learning

# install mamba in your base env -- recommended but not required for the next step. Just replace mamba with conda when we build the env from the environment.yml file

# activate the base env
conda activate
# install mamba
conda install -c conda-forge mamba

# build the env from the environment.yml file
mamba env create -n DL_PSI_env --file environment.yml

# activate the new env
conda activate DL_PSI_env
```

You can double check that PyTorch is set up to use your GPU with 
```
# activate the env -- if you haven't already
conda activate DL_PSI_env

# run the PyTorch debug script
python collect_env
```

And confirming that the output indicates that PyTorch thinks that CuDA is available.

# Running the Code:

## **Model Training**: 
---
### Hénon Map Experiments:

We recommend starting with replicating our Hénon map experiments. The config file is `configs/henon_study.cfg`. 

Be sure that the `num_workers` parameter in this file is set to the correct number of CPU cores on your machine (ideally the number of physical cores) and `num_gpu` is set to the number of GPUs that you want to use. There is likely no need to use more than 1 GPU for this experiment -- the small batch sizes that we use mean that you can probably train two models on the same GPU simultaneously (simply run two instances of a training script in different terinal windows -- with different configs so you aren't duplicating work of course).

Confirm where you want to save model checkpoints and logs by setting the `log_dir` parameter in the config file (default is to create a new nested directory `experiments/henon_map/` in the repo directory). The saved weights for a ResNet18 model are around **22 MB**, so you will need **~400 MB disk space** to store the top five models for each training set size, with and without data augmentation.

`configs/henon_study.cfg` will train models on different sized training sets, with and without data augmentation. This will probably take a few hours -- you can monitor the progress of your training runs with TensorBoard by running the command `tensorboard --logdir experiments/henon_map/` in another terminal window and going to the URL that is printed to the terminal in a web browser.

To train the models for the **Hénon map** system, run the following commands in a terminal window:

```
# cd into the repo
cd poincare_deep_learning

# activate the conda env
conda activate poincare_DL_env

# run the Hénon map experiments
python henon_map_train_model.py --cfg configs/henon_study.cfg
```

### Swinging Atwood's Machine Experiments:
If you haven't already, we recommend that you first try replicating our Hénon map experiments. 

To replicate our SAM experiments you will first need to create the dataset -- this process will occupy a consumer CPU for quite a few hours, and **will generate several GB of data** on your machine. 

After you've created the dataset, you'll want to be sure to set up `configs/SAM_study.cfg` and ensure that you update lines with `./datasets/` to include the correct path to the dataset you created on your machine. 

From here things are much the same as with the Hénon map dataset, although *the SAM dataset is larger and will be more RAM instensive*. Due to this you may not be able to run full scale experiments on a consumer setup, and may need dial down the `sample_frac` to load and use fewer samples during training.

<!-- Rather than directly splitting stored data into train/validation/test splits, we instead use lookup tables which are in turn split. These lookup tables tell us what .npz file the trajectory data is located in, as well as information about the sample (parameter value, as well as the corresponding logfile for the dataset generation run). -->

Be sure that the `num_workers` parameter in this file is set to the correct number of CPU cores on your machine (ideally the number of physical cores) and `num_gpu` is set to the number of GPUs that you want to use. There is likely no need to use more than 1 GPU for this experiment -- the small batch sizes that we use mean that you can probably train two models on the same GPU simultaneously (simply run two instances of a training script in different terinal windows -- with different configs so you aren't duplicating work of course).

Confirm where you want to save model checkpoints and logs by setting the `log_dir` parameter in the config file (default is to create a new nested directory `experiments/SAM/` in the repo directory). The saved weights for a ResNet18 model are around **22 MB**, so you will need **~400 MB disk space** to store the top five models for each training set size, with and without data augmentation.

`configs/henon_study.cfg` will train models on different sized training sets, with and without data augmentation. This will probably take a few hours -- you can monitor the progress of your training runs with TensorBoard by running the command `tensorboard --logdir experiments/henon_map/` in another terminal window and going to the URL that is printed to the terminal in a web browser.

To train the models for the **Swinging Atwood's Machine (SAM)** system, run the following commands in a terminal window:
```
# cd into the repo
cd poincare_deep_learning

# activate the conda env
conda activate poincare_DL_env

# run the SAM experiments
python SAM_train_model.py --cfg configs/SAM_study.cfg
```
    
## **Plotting Experiment Results**:

---
After running the respective training scripts, you can replicate the analysis from the paper by running the `SAM_analysis_plots.ipynb` and `hénon_map_analysis_plots.ipynb` notebooks. These notebooks will load configuration information from the respective config files, and will perform analysis on the saved models from the studies.

---

# Hardware Requirements:
<!-- --- -->
A GPU is more or less required to run the code in this repo. Our Hénon map experiments are runable on a consumer NVIDIA 10-series GPU with 8 GB of system RAM. Depending on your GPU model and installed NVIDIA drivers you may need to modify `environment.yml` to use a different version of PyTorch / CUDA.

Our SAM experiments require several GB of disk space to store trajectory data, as well as ~32GB of system RAM to run comfortably.

# Known issues:
- GTX 1660 Ti GPUs currently return NaNs when running with FP16 precision (see e.g. https://github.com/pytorch/pytorch/issues/58123). Switching to full FP32 precision (i.e. setting `precision = 32` in the `configs/` files allows our experiments to run, but may not reproduce the results in the paper.