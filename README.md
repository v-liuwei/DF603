# DF603 水实验队😎

## 快速开始

clone 仓库到本地

```bash
git clone https://github.com/v-liuwei/DF603.git
```

然后从[比赛网站](https://www.datafountain.cn/competitions/603/datasets)下载数据集, 解压后放到 `data` 目录下。

文件结构参考：
```text
DF603/
├── data/
|   ├── training_dataset/
|   |   ├── daily_dataset.csv
|   |   ├── hourly_dataset.csv
|   |   └── ...
|   ├── sample_submission.csv
|   └── test_public.csv
├── results/
|   ├── figs/
|   |   ├── *.png
|   |   └── ...
|   └── ...
└── src/
    ├── *.py
    └── ...
```

### 运行方式

```sh
.../DF603 $ pip install -r requirements.txt
.../DF603 $ cd src
.../DF603/src $ python darts_based.py
```

## 关键步骤

### 数据处理

- [x] 数据可视化, 见 `src/plot_data.py`
- [x] 异常值处理, 见 `src/preprocess_data.py`
- [x] 缺失值处理, 见 `src/preprocess_data.py`
- [x] 特征工程，见 `src/preprocess_data.py`

### 模型构建

- [x] 基于简单规则的方法，比如直接复制前一周的训练数据，见 `src/rule_based.py`
- [x] 基于机器学习的方法，如 LightGBM，见 `src/darts_based.py` (基于 `darts` 库) 和 `src/lightgbm_based` (基于 `lightgbm` 库)
- [ ] 基于深度学习的方法，如 LSTM
