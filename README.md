# DF603 æ°´å®éªéð

## å¿«éå¼å§

clone ä»åºå°æ¬å°

```bash
git clone https://github.com/v-liuwei/DF603.git
```

ç¶åä»[æ¯èµç½ç«](https://www.datafountain.cn/competitions/603/datasets)ä¸è½½æ°æ®é, è§£ååæ¾å° `data` ç®å½ä¸ã

æä»¶ç»æåèï¼
```text
DF603/
âââ data/
|   âââ training_dataset/
|   |   âââ daily_dataset.csv
|   |   âââ hourly_dataset.csv
|   |   âââ ...
|   âââ sample_submission.csv
|   âââ test_public.csv
âââ results/
|   âââ figs/
|   |   âââ *.png
|   |   âââ ...
|   âââ ...
âââ src/
    âââ *.py
    âââ ...
```

### è¿è¡æ¹å¼

```sh
.../DF603 $ pip install -r requirements.txt
.../DF603 $ cd src
.../DF603/src $ python darts_based.py
```

## å³é®æ­¥éª¤

### æ°æ®å¤ç

- [x] æ°æ®å¯è§å, è§ `src/plot_data.py`
- [x] å¼å¸¸å¼å¤ç, è§ `src/preprocess_data.py`
- [x] ç¼ºå¤±å¼å¤ç, è§ `src/preprocess_data.py`
- [x] ç¹å¾å·¥ç¨ï¼è§ `src/preprocess_data.py`

### æ¨¡åæå»º

- [x] åºäºç®åè§åçæ¹æ³ï¼æ¯å¦ç´æ¥å¤å¶åä¸å¨çè®­ç»æ°æ®ï¼è§ `src/rule_based.py`
- [x] åºäºæºå¨å­¦ä¹ çæ¹æ³ï¼å¦ LightGBMï¼è§ `src/darts_based.py` (åºäº `darts` åº) å `src/lightgbm_based` (åºäº `lightgbm` åº)
- [ ] åºäºæ·±åº¦å­¦ä¹ çæ¹æ³ï¼å¦ LSTM
