# Word-Segmentation
Prerequisite: <br>
1. ```pip install transformers==4.42.3```
2. ```pip install accelerate==0.32.1 -U```
3. ```pip install datasets```
4. ```pip install ckip-transformers==0.3.4```

Goal: To perform word segmentation and part-of-speech tagging for product names.
## ws_train.py
This file fine-tunes [the ws model](<https://github.com/ckiplab/ckip-transformers?tab=readme-ov-file>) from National Academy of Taiwan.
## ws_test.py
This file combines two models to perform word segmentation and part-of-speech tagging for product names. First, the product name is input into the tuned model, and then the output is fed into the POS model.
