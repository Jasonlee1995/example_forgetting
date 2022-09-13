# Example Forgetting Implementation with Pytorch
- Unofficial implementation of the paper *An Empirical Study of Example Forgetting during Deep Neural Network Learning*


## 0. Brief Explanation
- Check data with no learning events with various epochs - 15, 40, 60
- ResNet 50 mixed precision training on ImageNet-1K
- All the experiments are done with 1 RTX 3080 GPU


## 1. Implementation Details
- _eda : related with data pre-processing and eda
- _not_learned_json : images with no learning events
- _train_logs : train logs of ResNet 50 with mixed precision training
- data.py : data augmentations, dataset
- ImageNet_class_index.json : ImageNet-1K label information
- main_single.py : train and evaluate model and save forgetting event logs
- not_learned_train_filtered.json : pre-process not_learned_train.json using epochs
- not_learned_train.json : data - not learned epochs output
- run.sh : run python per gpu
- utils.py : utils such as scheduler, logger


## 2. Results
### 2.1. Base Configuration
|config|value|
|:-:|:-:|
|optimizer|SGD|
|base learning rate|0.05|
|weight decay|1e-4|
|optimizer momentum|0.9|
|batch size|128|
|learning rate schedule|cosine decay|
|warmup epochs|5|
|augmentation|Resize and Random Crop|
|loss|CrossEntropy|

### 2.2. Train Results
|epoch|acc|
|:-:|:-:|
|15|71.512|
|40|73.624|
|60|73.718|


## 3. Reference
- An Empirical Study of Example Forgetting during Deep Neural Network Learning [[paper](https://arxiv.org/abs/1812.05159)] [[official code](https://github.com/mtoneva/example_forgetting)]
