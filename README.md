# 太陽能發電量預測
###### tags: `Doc`

> 1. The complete workflow is shared at [Competition | AIdea太陽能發電量預測 — 競賽回顧
](https://medium.com/@waynechuang97/competition-aidea%E5%A4%AA%E9%99%BD%E8%83%BD%E7%99%BC%E9%9B%BB%E9%87%8F%E9%A0%90%E6%B8%AC-%E7%AB%B6%E8%B3%BD%E5%9B%9E%E9%A1%A7-78ef4679f0cd)

## Objective
Given environmental factors (e.g., irradiance, temperature), the metadata of solar power generators (e.g., generator capacity), etc., competitors are asked to predict the solar power generation (kWh) of generators located in different areas.

## How to Run
Following is the step-by-step guideline for generating the final result.
### *a. Data Preparation*
Run command `sh data_preparation/run.sh`<br>
> Output data `train.csv` and `test.csv` are dumped under path `data/processed/`.

### *b. Base Model Training*
Complete training process is configured by two configuration files, including `config/dp.yaml` and `config/model/<model_name>.yaml`.
`dp.yaml` controls data processing pipeline and `<model_name>.yaml` is the hyperparameter setting for the specified model.<br>
Base model is trained as follows (following argument setting is just an example):
#### 1. Configure `config/dp.yaml`
For more detailed information, please refer to [`dp_template.yaml`](https://github.com/JiangJiaWei1103/Solar-Power-Generation-Forecasting/blob/master/config/dp_template.yaml).
#### 2. Configure `config/<model_name>.yaml`
For more detailed information, please refer to [`config/model/`](https://github.com/JiangJiaWei1103/Solar-Power-Generation-Forecasting/tree/master/config/model).
#### 3. Train Base Model
Run command 
```
python -m tools.train_eval --project-name <project_name> --input-path ./data/processed/train.csv --model-name lgbm --cv-scheme gpts --n-folds 3 --oof-size 112 --group Date
```
For more detailed information about arguments, please run command `python -m tools.train_eval -h`<br>
Output structure is as follows:
```
    output/
       ├── config/ 
       ├── models/
       ├── trafos/
       ├── preds/
       ├── imp/
```
> All dumped objects are pushed to `WandB` remote.

### *c. Base Model Inference*
Final prediction can be obtained by running inference using pre-trained models (following argument setting is just an example):<br>
Run command
```
python -m tool.pred --project-name <project_name> --exp-id <exp_id> --input-path ./data/processed/test.csv --model-name lgbm --model-version <model_version> --model-type fold --model-ids 0 1 2
```
For more detailed information about arguments, please run command `python -m tools.pred -h`<br>
Output structure is as follows:
```
    ./submission.csv   # For quick submission
```
> The submission file is pushed to `WandB` remote.
