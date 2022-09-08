# Unconditional Audio Generation Benchmark
This is a fork from [Sashimi](https://github.com/HazyResearch/state-spaces/tree/main/sashimi), we borrow the automatic evaluation code for benchmarking different audio generators.

Please cite:
```
@article{goel2022sashimi,
  title={It's Raw! Audio Generation with State-Space Models},
  author={Goel, Karan and Gu, Albert and Donahue, Chris and R{\'e}, Christopher},
  journal={International Conference on Machine Learning ({ICML})},
  year={2022}
}
```

# Automated Metrics
A standalone implementations of automated evaluation metrics for evaluating the quality of generated samples on the SC09 dataset in `metrics.py`. Following [Kong et al. (2021)](https://arxiv.org/pdf/2009.09761.pdf), the metrics and the procedure is defined as follows:
## Fréchet Inception Distance (FID)

uses the classifier to compare moments of generated and real samples in feature space.

## Inception Score (IS) 
IS measures the quality of the generated data and detects whether there is a mode collapse by using a pretrained domain-specific classifier.  It is computed as the KL divergence between the conditional probability p(y|x) and marginal probability p(y),

## Modified Inception Score (mIS)
provides a measure of both intra-class in addition to inter-class diversity.
## AM Score
uses the marginal label distribution of training data compared to IS.
## Number of statistically different bins score (NDB)
the number of bins that contain statistically different proportion of samples between training samples and generated samples. 

## SC09 Classifier Training
A modified version of the training/testing script provided by the [pytorch-speech-commands](https://github.com/tugstugi/pytorch-speech-commands) repository, following [Kong et al. (2021)](https://arxiv.org/pdf/2009.09761.pdf).

This classifier has two purposes:
1. To calculate the automated metrics, each SC09 audio clip must be converted into a feature vector. 
2. Following [Donahue et al. (2019)](https://arxiv.org/pdf/1802.04208.pdf), a classifier confidence is used as a proxy for the quality and intelligibility of the generated audio. Roughly, a large number of samples from each model are sampled, and then the top samples are selected (as ranked by classifier confidence) per class (as assigned by the classifier).

### Install Dependencies
Requirements are included in the `requirements.txt` file for reference excluding `torch` and `torchvision`. (It's been tested with `torch` version `1.9.0+cu102`.)
```bash
pip install -r requirements.txt
```

### Download Dataset
For convenience, it is recommend redownloading the Speech Commands dataset for classifier training using the commands below. Downloading and extraction should take a few minutes.
```bash
cd ./sc09_classifier/
bash download_speech_commands_dataset.sh
```

### Checkpoint
To train the classifier from the scratch, run the following command:
```bash
mkdir checkpoints/
python train_speech_commands.py --batch-size 96 --learning-rate 1e-2
```
The best model checkpoint should be saved under `./sc09_classifier/` with a leading timestamp. 

It's also possible to reuse the classifier checkpoint and cache files provided by the authors directly on the [Huggingface Hub](https://huggingface.co/krandiash/sashimi-release) at `sc09_classifier/resnext.pth`. This model achieves `98.08%` accuracy on the SC09 test set. Download this checkpoint and cache file and place place them in `./sc09_classifier/`.

At the end of this your directory structure should look something like this:
```bash
samples/
├── sc09/
sc09_classifier/
├── resnext.pth
...
```

## Calculating Automated Metrics
Instructions for calculating the automated SC09 metrics 

### Metrics on Non-Autoregressive Models
For generators which don't provide exact likelihoods, we can simply calculate metrics directly on 2048 samples
```
python test_speech_commands.py --sample-dir /samples/sc09/2048-diffwave/ resnext.pth
```

### Metrics on Autoregressive Models (DiffWave variants, WaveGAN etc)
For autoregressive models, we can follow a threshold tuning procedure: first, we generated 10240 samples for each model, using 5120 to tune thresholds for rejecting samples with the lowest and highest likelihoods, and evaluating the metrics on the 5120 samples that are held out. This is all taken care of automatically by the test_speech_commands.py script (with the --threshold flag passed in).
```
# SaShiMi (4.1M parameters)
python test_speech_commands.py --sample-dir /samples/sc09/10240-sashimi/ --threshold resnext.pth
```

### Dataset Metrics
To generate the automated metrics for the dataset, run the following command:
```bash
python test_speech_commands.py resnext.pth
```
If you didn't correctly place the `cache` folder under `./sc09_classifier`, this will be a little slow to run the first time, as it caches features and predictions (`train_probs.npy`, `test_probs.npy`, `train_activations.npy`, `test_activations.npy`) for the train and test sets under `./sc09_classifier/cache/`. Subsequent runs reuse this and are much faster.

### Examples
| System    |  Generator   | FID | Inception  | mInception | AM | 
|-----------|----|-----|------------|---------|--------|
| wavenet   | AR   | 4.93  |2.39      | 6.06     |1.45  |
| samplernn | AR   | 8.97 |1.71      | 2.98      |1.77  |
|sashimi    | AR   |  2.03| 4.31     | 25.88   | 0.88  |
|[diffwave](https://github.com/philsyn/DiffWave-unconditional)  | diffusion|1.80|5.70|51.88|0.65|

* Note: AR models here are based on the generated results from s4 v2 package. Diffwave is based on the provided pretrained model.

# References:
[1] S4 github: https://github.com/HazyResearch/state-spaces
[2] Tun-Min Hung et al. A Benchmarking Initiative for Audio-domain Music Generation using the {FreeSound Loop Dataset}

# TODO

- [ ] Add FAD
- [ ] Add results for diffusion based generators
