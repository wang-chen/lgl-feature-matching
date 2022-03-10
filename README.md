# Feature matching with FGN

This repo contains the source code for the feature matching application (Sec. 7) in ["Lifelong Graph Learning." Chen Wang, Yuheng Qiu, Dasong Gao, Sebastian Scherer. *CVPR 2022*.]((https://arxiv.org/abs/2009.00647))

## Usage
### Dependencies

 - Python >= 3.5
 - PyTorch >= 1.7
 - OpenCV >= 3.4
 - NumPy
 - TensorBoard
 - Matplotlib
 - ArgParse
 - tqdm

### Data
The [TartanAir](https://theairlab.org/tartanair-dataset/) dataset is required for both training and testing. The dataset should be aranged as follows:
```
$DATASET_ROOT/
└── tartanair/
    ├── abandonedfactory_night/
    └── ...
```

### Commandline
Training and evaluates the method with default setting:
```sh
$ python train.py --data-root <DATASET_ROOT> --method <FGN/GAT>
```
- `--method` option is used to switch between FGN-based (ours) and GAT-based (SuperGlue) graph matcher
- Considering the gigantic volume of TartanAir, evaluation will happen every 5000 training steps by default (can be overriden by `--eval-freq`). Results will be logged to the console.
- If `--log-dir` is specified, TensorBoard will be activated to show visualization and evaluation results instead (under "TEXT" tab).
- Detailed description of settings can be viewed by `$ python train.py -h`.

## Citation
```bibtex
@inproceedings{wang2022lifelong,
  title={Lifelong graph learning},
  author={Wang, Chen and Qiu, Yuheng and Gao, Dasong and Scherer, Sebastian},
  booktitle={2022 Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
