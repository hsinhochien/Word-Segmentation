# Word-Segmentation
Prerequisite: <br>
1. ```pip install transformers==4.42.3```
2. ```pip install accelerate==0.32.1 -U```
3. ```pip install datasets```

Goal: To segment product names.
## ws_train.py
This file fine-tunes [the model](<https://huggingface.co/ckiplab/bert-base-chinese-ws>) from HuggingFace.
## ws_test.py
This file demonstrates how to use the tuned model directly and shows the results.
