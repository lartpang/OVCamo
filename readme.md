# (ECCV 2024) Open-Vocabulary Camouflaged Object Segmentation

<p align="center">
   <a href='https://arxiv.org/abs/2311.11241'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv PDF'>
   </a>
   <br/>
   <img src='https://github.com/lartpang/OVCamo/assets/26847524/d2c474f2-4bde-455c-af71-e0761e57a574' alt='logo'>
</p>


```bibtex
@inproceedings{OVCOS_ECCV2024,
  title={Open-Vocabulary Camouflaged Object Segmentation},
  author={Pang, Youwei and Zhao, Xiaoqi and Zuo, Jiaming and Zhang, Lihe and Lu, Huchuan},
  booktitle=ECCV,
  year={2024},
}
```

> [!note]
>
> Details of the proposed OVCamo dataset can be found in the document for [our dataset](https://github.com/lartpang/OVCamo/releases/download/dataset-v1.0/ovcamo.zip).

## Prepare Dataset

![image](https://github.com/lartpang/OVCamo/assets/26847524/92f5f7e8-55a9-4d7e-bc41-264d255af658)

1. Prepare the training and testing splits: See the document in [our dataset](https://github.com/lartpang/OVCamo/releases/download/dataset-v1.0/ovcamo.zip) for details.
2. Set the training and testing splits in the yaml file `env/splitted_ovcamo.yaml`:
   - `OVCamo_TR_IMAGE_DIR`: Image directory of the training set.
   - `OVCamo_TR_MASK_DIR`: Mask directory of the training set.
   - `OVCamo_TR_DEPTH_DIR`: Depth map directory of the training set. Depth maps of the training set which are generated by us, can be downloaded from <https://github.com/lartpang/OVCamo/releases/download/dataset-v1.0/depth-train-ovcoser.zip>
   - `OVCamo_TE_IMAGE_DIR`: Image directory of the testing set.
   - `OVCamo_TE_MASK_DIR`: Mask directory of the testing set.
   - `OVCamo_CLASS_JSON_PATH`: Path of the json file `class_info.json` storing class information of the proposed OVCamo.
   - `OVCamo_SAMPLE_JSON_PATH`: Path of the json file `sample_info.json` storing sample information of the proposed OVCamo.

## Training/Inference

1. Install dependencies: `pip install -r requirements.txt`.
   1. The versions of `torch` and `torchvision` are listed in the comment of  `requirements.txt`.
2. Run the script to:
   1. train the model: `python .\main.py --config .\configs\ovcoser.py --model-name OVCoser`;
   2. inference the model: `python .\main.py --config .\configs\ovcoser.py --model-name OVCoser --evaluate --load-from <path of the local .pth file.>`.

## Evaluate the Pretrained Model

1. Download [the pretrained model](https://github.com/lartpang/OVCamo/releases/download/model-v1.0/model.pth).
2. Run the script: `python .\main.py --config .\configs\ovcoser.py --model-name OVCoser --evaluate --load-from model.pth`.

## LICENSE

- Code: [MIT LICENSE](./LICENSE)
- Dataset: <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/lartpang/OVCamo">OVCamo</a> by <span property="cc:attributionName">Youwei Pang, Xiaoqi Zhao, Jiaming Zuo, Lihe Zhang, Huchuan Lu</span> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
