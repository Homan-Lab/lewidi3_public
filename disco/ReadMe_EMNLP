# DisCo: Distribution from Context

This repository contains the code to reproduce the experiments from DisCo (Distribution from Context), a neural model that learns to predict label distributions by jointly modeling item-level and annotator-level label distributions.

## Abstract

Annotator disagreement is common whenever human judgment is needed for supervised learning. It is conventional to assume that one label per item represents ground truth. However, this obscures minority opinions, if present. We regard "ground truth″ as the distribution of all labels that a population of annotators could produce, if asked (and of which we only have a small sample). We next introduce DisCo (Distribution from Context), a simple neural model that learns to predict this distribution. The model takes annotator-item pairs, rather than items alone, as input, and performs inference by aggregating over all annotators. Despite its simplicity, our experiments show that, on six benchmark datasets, our model is competitive with, and frequently outperforms, other, more complex models that either do not model specific annotators or were not designed for label distribution learning.

## Recent Improvements (LeWiDi-2025)

Our recent work at LeWiDi-2025 extends DisCo with several key improvements:

1. **Annotator Metadata Embeddings**: Enhanced input representations by incorporating annotator demographic and contextual metadata using sentence transformers
2. **Multi-Objective Training Losses**: Improved loss functions that directly optimize for distributional metrics (Wasserstein distance, mean absolute distance) alongside traditional KL divergence
3. **Loss Reweighting**: Flexible loss combination strategies (soft label, perspectivist, combined, alternating) for better disagreement modeling

These improvements yield substantial gains in both soft label and perspectivist evaluation metrics across multiple datasets.

## Features

- **Joint Modeling**: DisCo jointly models item-level, annotator-level, and aggregate label distributions
- **Metadata Support**: Incorporates annotator metadata (demographics, education, etc.) through learned embeddings
- **Autonomous Preprocessing**: LLM-powered preprocessing agent that automatically handles diverse dataset formats
- **Flexible Loss Functions**: Multiple objective functions optimized for different evaluation scenarios
- **WandB Integration**: Full experiment tracking and hyperparameter tuning via Weights & Biases

## Installation

### Requirements
Install all rrequirements via requirments.txt

### Setup WandB

1. Create a free account at [https://wandb.ai](https://wandb.ai)
2. Get your API key from the settings page
3. Update `wandb_creds.py` with your credentials:

```python
def wandb_creds():
    return "your-wandb-api-key-here"

def wandb_entity_value():
    return "your-wandb-username"
```

## Quick Start

### 1. Prepare Your Dataset

For specific datasets, use the dedicated preprocessing scripts:

```bash
# For MP dataset
python3 preprocess_mp.py --> run this for the necessary train/test and val files change path inside code

# For CSC dataset  
python3 preprocess_csc.py  --> run this for the necessary train/test and val files change path inside code

# For PP dataset
python3 preprocess_ppp_test.py 
python3 preprocess_ppp_train.py 
python3 preprocess_ppp_dev.py 
```

### 2. Generate DisCo Dataset Format

Convert preprocessed data to DisCo's internal format:

```bash
python3 gen_disco_dataset.py \
    --inp_dir=./datasets/your_dataset/processed/ \
    --out_dir=./experimental_data/your_dataset/ \
    --annotator_item_fname=your_dataset_train_AIL.csv \
    --item_lab_fname=your_dataset_train_IL.csv \
    --annotator_lab_fname=your_dataset_train_AL.csv \
    --embeddings=Xi_train.npy \
    --split_name=train
```

Repeat for `dev` and `test` splits.


### 3. Create WandB Sweep

1. Go to your WandB project and click **Sweeps** → **Create Sweep**
2. Use the following YAML configuration:

```yaml
method: random
metric:
  goal: minimize
  name: train KL
parameters:
  act_fx:
    values:
    - softsign
    - tanh
    - relu
    - relu6
    - elu
    - swish
  drop_p:
    values:
    - 0.5
    - 0.75
    - 0.3
    - 0.2
    - 0
  gamma_a:
    value: 1
  gamma_i:
    value: 1
  lat_a_dim:
    value: 256
  lat_dim:
    value: 512
  lat_fusion_type:
    value: concat
  lat_i_dim:
    values:
    - 128
    - 256
    - 512
    - 1024
  learning_rate:
    distribution: constant
    value: 0.001
  opt_type:
    values:
    - adam
  update_radius:
    value: -2
  weight_init_scheme:
    values:
    - gaussian
    - orthogonal
    - uniform
    - xavier_normal
  meta_dim:
    value: 768
```

3. Copy the sweep ID (e.g., `your-project/sweep-name/abc123`)

### 5. Train the Model

```bash
python3 train_disco_sweep.py \
    --config ./config_files/your_dataset.cfg \
    --sweep_id your-project/sweep-name/abc123 \
    --gpu_id 0
```

### 6. Evaluate the Model

```bash
python3 eval_model.py \
    --data_dir=./experimental_data/your_dataset/ \
    --model_fname=./experimental_data/your_dataset/trained_model.disco \
    --split_name=test \
    --dataset_name=your_dataset \
    --wandb_name=your_project \
    --empirical_fname=./datasets/your_dataset/processed/disco/your_dataset_test_AIL_data.csv \
    --gpu_id=0
```

## Dataset Format

### Input Format

DisCo expects datasets in JSON format with the following structure:

```json
{
  "item_id_1": {
    "text": {
      "field1": "text content 1",
      "field2": "text content 2"
    },
    "annotations": {
      "annotator_1": "label_1",
      "annotator_2": "label_2"
    },
    "soft_label": {
      "label_1": 0.6,
      "label_2": 0.4
    },
    "split": "train"
  }
}
```

### Metadata Format

Annotator metadata should be a JSON file mapping annotator IDs to metadata:

```json
{
  "annotator_1": {
    "Age": 25,
    "Gender": "Female",
    "Nationality": "US",
    "Education": "Bachelor"
  }
}

make sure to do the necessary changes in the helper_functions.py where you can change the natural language structure for the necessary data (last function in the file)
```

## Model Architecture

DisCo consists of:

1. **Item Encoder**: Projects item feature vectors (e.g., sentence embeddings) to latent space
2. **Annotator Encoder**: Projects annotator metadata embeddings to latent space  
3. **Fusion Layer**: Combines item and annotator representations (concat or sum)
4. **Transformation Layer**: Applies non-linear transformations with dropout
5. **Decoders**: Three separate decoders for:
   - Item-level label distribution (`yi`)
   - Annotator-level label distribution (`ya`)
   - Aggregate label distribution (`y`)

## Key Improvements

### Annotator Metadata Embeddings

DisCo now incorporates annotator metadata (demographics, education, etc.) using sentence transformers. Metadata is converted to descriptive sentences and embedded using `paraphrase-multilingual-mpnet-base-v2`.

### Multi-Objective Loss Functions

The model supports multiple loss objectives:

- **Original**: KL divergence for all distributions
- **Soft Label**: Optimizes for Wasserstein distance (soft label evaluation)
- **Perspectivist**: Optimizes for mean absolute distance (perspectivist evaluation)
- **Combined**: Weighted combination of soft and perspectivist losses


## Supported Datasets

The codebase includes preprocessing scripts for:

- **CSC**: Sarcasm detection dataset
- **PP**: Paraphrase detection dataset
- **MP**: Multi-perspective dataset

## Project Structure

```
disoc_ann/
├── model/
│   └── disco.py              # DisCo model implementation
├── utils/
│   ├── config.py             # Configuration parsing
│   └── utils.py              # Utility functions
├── config_files/             # Dataset configuration files
├── datasets/                 # Dataset storage
│   └── [dataset_name]/
│       ├── processed/        # Preprocessed data
│       └── [dataset]_metadata.json
├── experimental_data/        # Training outputs
├── autonomous_data_agent_llm.py  # LLM-powered preprocessing
├── gen_disco_dataset.py      # Dataset format conversion
├── train_disco_sweep.py      # Training script
├── eval_model.py             # Evaluation script
├── predict_disco.py          # Prediction script
└── run_disco_autonomous.sh   # Complete pipeline script
```

## Hyperparameter Tuning

DisCo uses WandB Sweeps for hyperparameter tuning. Key hyperparameters:

- **Architecture**: `lat_i_dim`, `lat_a_dim`, `lat_dim`, `lat_fusion_type`
- **Regularization**: `gamma_i`, `gamma_a`, `drop_p`, `l1_norm`, `l2_norm`
- **Optimization**: `learning_rate`, `opt_type`, `update_radius`
- **Activation**: `act_fx` (relu, tanh, swish, etc.)

See the WandB Sweeps section above for a complete configuration template.

## Citation

If you use DisCo in your research, please cite:

### LeWiDi-2025 Extension
```
@inproceedings{sawkar-etal-2025-lpi,
    title = "LPI-RIT at LeWiDi-2025: Improving Distributional Predictions via Metadata and Loss Reweighting with DisCo",
    author = "Sawkar, Mandira and
              Shetty, Samay U. and
              Pandita, Deepak and
              Weerasooriya, Tharindu Cyril and
              Homan, Christopher M.",
    booktitle = "Proceedings of the The 4th Workshop on Perspectivist Approaches to NLP",
    month = {November},
    year = {2025},
    address = {Suzhou, China},
    publisher = {Association for Computational Linguistics},
    pages = {196--207},
    url = {https://aclanthology.org/2025.nlperspectives-1.17}
}
```

### Original DisCo Paper (ACL 2023)
```
@inproceedings{weerasooriya-etal-2023-disagreement,
    address = {Toronto, Canada},
    author = {Weerasooriya, Tharindu Cyril and
              Alexander G. Ororbia II and
              Bhensadadia, Raj and
              KhudaBukhsh, Ashiqur and
              Homan, Christopher M.},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
    month = {July},
    pages = {4679--4695},
    publisher = {Association for Computational Linguistics},
    title = {Disagreement Matters: Preserving Label Diversity by Jointly Modeling Item and Annotator Label Distributions with DisCo},
    url = {https://aclanthology.org/2023.findings-acl.287},
    year = {2023}
}
```

## License

[Add your license information here]

## Contact

For questions or issues, please open an issue on GitHub or email - ss4711@rit.edu

## Acknowledgments

We thank the LeWiDi-2025 shared task organizers and the DisCo collaborators for their contributions.

