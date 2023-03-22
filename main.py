from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import pickle, re, itertools, pandas as pd
from transformers import pipeline, set_seed, PhobertTokenizer, AutoModelForMaskedLM
from underthesea import text_normalize
from pyvi import ViTokenizer, ViUtils, ViPosTagger
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import asyncio
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
import uvicorn,ray
import os

ray.init(dashboard_host='192.168.1.5')

origins = ["*"]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('Sentiment/sentiment_model_bi-lstm_cnn.h5')

with open('Sentiment/tokenizer_data.plk', 'rb') as input_file:
    my_tokenizer = pickle.load(input_file)


async def load_mlm_model():
    tokenizer = PhobertTokenizer.from_pretrained("Sentiment/models")
    model = AutoModelForMaskedLM.from_pretrained("Sentiment/models")
    return tokenizer, model


async def preprocessingData(input):
    newInput = text_normalize(input)  # chuẩn hóa dấu từ
    newInput = newInput.lower()  # chuyển về chữ thường
    newInput = newInput.replace('\n', '. ')  # thay thế xuống dòng bằng dấu "."
    newInput = newInput.strip()  # xóa khoảng trắng đầu và cuối chuỗi
    newInput = re.sub(r'[^\w\s]', '', newInput)  # xóa kí tự đặc biệt
    newInput = re.sub(r'[0-9]', '', newInput)  # xóa số
    newInput = re.sub(' +', ' ', newInput)  # thay thế nhiều khoảng trắng thành một khoảng trắng
    newInput = newInput.replace(' .', '.')  # xóa khoảng trắng trc dấu "." ở cuối câu

    return newInput


async def clean_emoji(input):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)

    input = re.sub(emoj, '', input)  # Xóa biểu tượng cảm xúc
    return input


async def clean_acronyms(input):
    replace = {'ko': 'không',
               'đc': 'được',
               'or': 'hoặc',
               'yo': 'yêu',
               'ns': 'nói',
               'ms': 'mới',
               'nchung': 'nói chung',
               'sp': 'sản phẩm',
               'kh': 'không',
               'mn': 'mọi người',
               'good': 'tốt'}

    words = input.split()
    for i, word in enumerate(words):
        if word in replace:
            words[i] = replace[word]

    return " ".join(words)


async def clean_duplicate(input):
    return ''.join(c[0] for c in itertools.groupby(input))


async def clean_stopword(input):
    text = []
    stop_words = pd.read_csv('Sentiment/vietnamese_stopword.txt')
    tmp = input.split(' ')
    for stop_word in stop_words:
        if stop_word in tmp:
            tmp.remove(stop_word)
    text.append(" ".join(tmp))

    return ' '.join(text)


async def tokenizer_word(input):
    input = ViTokenizer.tokenize(input)
    return input


async def count_mask(text_masked):
    split_masked_text = text_masked.split(" ")
    count_mask = 0

    for word in split_masked_text:
        if word == "<mask>":
            count_mask += 1

    return count_mask


def mlm(text_masked, model, tokenizer):
    # API_URL = "https://api-inference.huggingface.co/models/vinai/phobert-large"
    # headers = {"Authorization": "Bearer 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"}
    # response = requests.post(API_URL, headers=headers, json=text_masked).json()
    # mlm_result = response[0]["sequence"]
    # return mlm_result.lower()
    # print(text_masked)
    classifier = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    mlm_result = classifier(text_masked)[0]
    # print(text_masked)
    # print(mlm_result)
    # print(mlm_result.__class__.__name__)
    return mlm_result.get('sequence').lower()


async def list_mlm(text_masked,model, tokenizer):
    classifier = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    mlm_result = classifier(text_masked)[0]
    return mlm_result[0]


async def full_mlm(text_masked, model, tokenizer):
    get_count_masked = await count_mask(text_masked)
    if get_count_masked == 1:
        return mlm(text_masked,model=model,tokenizer=tokenizer)
    elif get_count_masked > 1:
        res_list_mlm = await list_mlm(text_masked,model=model, tokenizer=tokenizer).get('sequence').lower()
        return full_mlm(res_list_mlm, model=model, tokenizer=tokenizer)


async def pos(input, model, tokenizer):
    # print(input)
    token, label = ViPosTagger.postagging(input)
    # print(token)
    # print(label)
    # giữ lại danh từ(N,Np), trạng từ(R), động từ(V)
    res = []
    have_x = False
    for index, item in enumerate(label):
        if (item == 'N' or item == 'Np' or item == 'Ny' or item == 'Nc' or item == 'Nu'
                or item == 'V' or item == 'R' or item == 'A'):
            # res += token[index] + " "
            res.append(token[index])
        elif (item == 'X'):
            have_x = True
            res.append("<mask>")

    if (have_x):
        return full_mlm(" ".join(res), model=model, tokenizer=tokenizer)
    else:
        return " ".join(res)


async def handle_negation(text):
    negation_words = ['không', 'chưa', 'chẳng', 'chả', 'hết', 'không có', 'không thể', 'không nên', "tệ", "xấu"]
    positive_words = ['tốt', 'tuyệt', 'đẹp', 'hợp', 'ổn', 'xinh', 'ngon', 'hoàn hảo']
    negation_flag = False
    words = text.split()
    i = 0
    while i < len(words):
        if words[i] in negation_words:
            if i < len(words) - 1:
                if words[i + 1] in positive_words:
                    words[i] = "không tích cực"
                    words.remove(words[i + 1])
                    i += 2
                elif words[i + 1] in negation_words:
                    words[i] = "không tiêu cực"
                    words.remove(words[i + 1])
                    i += 2
                else:
                    i += 1
            elif i == len(words) - 1:
                if words[i] in positive_words:
                    words[i] = "không tích cực"
                elif words[i] in negation_words:
                    words[i] = "không tiêu cực"
                break

        else:
            i += 1

    return ' '.join(words)


async def pre_raw_input(raw_input, tokenizer):
    tokenizer_MLM, model_MLM = await load_mlm_model()

    input_text_pre = await preprocessingData(raw_input)  # chuẩn hóa cơ bản (xóa dấu câu, kí tự đặc biệt,...)
    input_text_pre = await clean_emoji(input_text_pre)  # xóa biểu tượng cảm xúc
    input_text_pre = await clean_duplicate(input_text_pre)  # xóa kí tự trùng lặp
    input_text_pre = await clean_acronyms(input_text_pre)  # chuẩn hóa từ viết tắt
    input_text_pre_accent = await tokenizer_word(input_text_pre)  # phân tách từ
    input_text_pre_accent = await clean_stopword(input_text_pre_accent)  # xóa stopp words
    input_text_pre_accent = await pos(input_text_pre_accent, model=model_MLM, tokenizer=tokenizer_MLM)  # pos tagging

    input_text_pre_accent = await handle_negation(input_text_pre_accent)  # xử lí phủ định

    tokenized_text = tokenizer.texts_to_sequences([input_text_pre_accent])
    vec_data = pad_sequences(tokenized_text, padding='post', maxlen=2500)

    return vec_data


async def inference_model(input, model):
    output = model(input).numpy()[0]
    result = output.argmax()
    conf = float(output.max())
    label_dict = {'NEG': 0, 'NEU': 1, 'POS': 2}
    label = list(label_dict.keys())
    return label[int(result)], conf


async def prediction(raw_input, tokenizer, model):
    # input_text_pre_accent = tokenizer_word(raw_input)  # phân tách từ
    # tokenized_text = tokenizer.texts_to_sequences([input_text_pre_accent])
    # vec_data = pad_sequences(tokenized_text, padding='post', maxlen=2500)
    # tokenizer_data = build_vocabulary()[7]
    vec_data = await pre_raw_input(raw_input, tokenizer)
    result, conf = await inference_model(vec_data,model)

    return result, conf


@app.get("/")
async def get_model():
    return {"check connect: ": "OK"}


@app.post("/predict")
async def predict(input:str):
    sentiment, accuracy = await prediction(input, tokenizer= my_tokenizer, model= model)
    return {"sentiment: ": sentiment, "accuracy:": accuracy}
    # return sentiment

if __name__ == '__main__':
    # print(asyncio.run(
    #     prediction(r"Pô ok nghe nổ lực nhưng sơn đen bị dính lên miếng ixon pô giá rẻ.", my_tokenizer, model)))
    # print(asyncio.run(predict("Cái này giá ổn không nhỉ?")))
    # uvicorn.run(app, host="192.168.1.5", port= 9000)
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run(app, host="192.168.1.5", port=port)


