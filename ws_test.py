# pip install ckip-transformers==0.3.4

from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import logging

FORMAT = '%(asctime)s [%(levelname)s]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

# Initialize drivers
logging.info("Initializing drivers ... WS")
ws_driver = CkipWordSegmenter(model_name="./best_model")#(model="bert-base")
logging.info("Initializing drivers ... POS")
pos_driver = CkipPosTagger(model="bert-base")

text = [
    "妙管家抗菌有酵洗衣精4000g x4入(箱)",
    "FIDOS》蚤蜱除噴劑250ml有效驅離跳蚤"
]

logging.info("Running pipeline ... WS")
ws = ws_driver(text)
logging.info("Running pipeline ... POS")
pos = pos_driver(ws)
print()

def pack_ws_pos_sentece(sentence_ws, sentence_pos):
    assert len(sentence_ws) == len(sentence_pos)
    res = []
    for word_ws, word_pos in zip(sentence_ws, sentence_pos):
        res.append(f"{word_ws}({word_pos})")
    return "\u3000".join(res)

for sentence, sentence_ws, sentence_pos in zip(text, ws, pos):
    print(sentence)
    print(pack_ws_pos_sentece(sentence_ws, sentence_pos))
    print()