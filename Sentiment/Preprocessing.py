import re, requests, pickle
from tensorflow.keras.utils import plot_model
from pyvi import ViTokenizer, ViUtils, ViPosTagger
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import initializers
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM, GRU, Input, GlobalMaxPooling1D, LayerNormalization, Conv1D, MaxPooling1D, Concatenate, Attention
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Sequential, Model, callbacks
import tensorflow as tf
import pandas as pd
import itertools
import numpy as np
from underthesea import text_normalize
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForMaskedLM
import matplotlib.pyplot as plt
from Translator import Translate
import asyncio
from sklearn.model_selection import train_test_split
import sklearn

async def load_mlm_model():
    tokenizer = AutoTokenizer.from_pretrained("./models")
    model = AutoModelForMaskedLM.from_pretrained("./models")
    return tokenizer, model


def preprocessingData(input):
    newInput = text_normalize(input)  # chuẩn hóa dấu từ
    newInput = newInput.lower()  # chuyển về chữ thường
    newInput = newInput.replace('\n', '. ')  # thay thế xuống dòng bằng dấu "."
    newInput = newInput.strip()  # xóa khoảng trắng đầu và cuối chuỗi
    newInput = re.sub(r'[^\w\s]', '', newInput)  # xóa kí tự đặc biệt
    newInput = re.sub(r'[0-9]', '', newInput)  # xóa số
    newInput = re.sub(' +', ' ', newInput)  # thay thế nhiều khoảng trắng thành một khoảng trắng
    newInput = newInput.replace(' .', '.')  # xóa khoảng trắng trc dấu "." ở cuối câu

    return newInput


def clean_emoji(input):
    positive_emoji = re.compile("["
                      u"\U0001F600"
                    u"\U0001F601"
                    u"\U0001F602"
                    u"\U0001F603"
                    u"\U0001F604"
                    u"\U0001F605"
                    u"\U0001F606"
                    u"\U0001F609"
                    u"\U0001F60A"
                    u"\U0001F60B"
                    u"\U0001F60C"
                    u"\U0001F60D"
                    u"\U0001F60E"
                    u"\U0001F60F"
                    u"\U0001F618"
                    u"\U0001F617"
                    u"\U0001F619"
                    u"\U0001F61A"
                    u"\U0001F642"
                    u'\u2764\ufe0f'
                    u'\U0001F917'
                    u'\U0001F92A'
                    u'\U0001F44D'
                    u'\U0001F609'
                    u'\U0001f495'
                    u'\U0001F60E'
                    u'\U0001F929'
                    u'\U0001F44D\U0001F3FB'
                    u'\U0001F44C'
                    u'\U0001F924'
                    u'\U0001F917'
                    u'\U0001F606'
                    u'\U0001F92D'
                    u'\U0001F496'
                    u'\U0001F636'
                    u'\U0001F609'
                    u'\U0001F60C'
                    u'\U0001F4AF'
                    u'\U0001F914'
                    u'\U0001F923'
                    u'\U0001F642'
                    u'\U0001F929'
                    u'\U0001F493'
                    u'\U0001F61D'   
                    u'\U0001F61D'
                    
                      "]+", re.UNICODE)

    negative_emoji = re.compile("["
                      u"\U0001F612"
                        u"\U0001F613"
                        u"\U0001F614"
                        u"\U0001F615"
                        u"\U0001F616"
                        u"\U0001F61E"
                        u"\U0001F61F"
                        u"\U0001F621"
                        u"\U0001F622"
                        u"\U0001F623"
                        u"\U0001F625"
                        u"\U0001F626"
                        u"\U0001F627"
                        u"\U0001F628"
                        u"\U0001F629"
                        u"\U0001F62A"
                        u"\U0001F62B"
                        u"\U0001F62D"
                        u"\U0001F641"
                        u'\U0001F624'
                        u'\U0001F611'
                        u'\U0001F616'
                        u'\U0001F92C'
                        u'\U0001F611'
                        u'\U0001F630'
                        u'\U0001F621'
                        u'\U0001F643'
                        u'\U0001F61D'
                        u'\U0001F61D'
                        u'\U0001F620'
                      "]+", re.UNICODE)

    neu_emoji = re.compile("["
                        u'\U0001F610'
                      "]+", re.UNICODE)


    input = re.sub(positive_emoji, 'tích cực', input)
    input = re.sub(negative_emoji, 'tiêu cực', input)# Xóa biểu tượng cảm xúc
    input = re.sub(neu_emoji, 'bình thường', input)
    return input


def clean_acronyms(input):
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
               'good': 'tốt',
               'sz': 'kích cỡ',
               'nt': 'nhắn tin',
               'trl': 'trả lời',
               'k': 'không',
                'size': 'kích cỡ',
               'mng': 'mọi người',

               }

    words = input.split()
    for i, word in enumerate(words):
        if word in replace:
            words[i] = replace[word]

    return " ".join(words)


def clean_duplicate(input):
    return ''.join(c[0] for c in itertools.groupby(input))


def clean_stopword(input):
    text = []
    stop_words = pd.read_csv('vietnamese_stopword.txt')
    tmp = input.split(' ')
    for stop_word in stop_words:
        if stop_word in tmp:
            tmp.remove(stop_word)
    text.append(" ".join(tmp))

    return ' '.join(text)


def tokenizer_word(input):
    input = ViTokenizer.tokenize(input)
    return input

def count_mask(text_masked):
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


def list_mlm(text_masked,model, tokenizer):
    classifier = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    mlm_result = classifier(text_masked)[0]
    return mlm_result[0]


def full_mlm(text_masked, model, tokenizer):
    get_count_masked = count_mask(text_masked)
    if get_count_masked == 1:
        return mlm(text_masked,model=model,tokenizer=tokenizer)
    elif get_count_masked > 1:
        res_list_mlm = list_mlm(text_masked,model=model, tokenizer=tokenizer).get('sequence').lower()
        return full_mlm(res_list_mlm, model=model, tokenizer=tokenizer)


def pos(input, model, tokenizer):
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


def pos_tagging(input):
    print(ViPosTagger.postagging(input))


def pr(input):
    input_text_pre = list(tf.keras.preprocessing.text.text_to_word_sequence(input))
    input_text_pre = " ".join(input_text_pre)
    print(input_text_pre)
    input_text_pre_no_accent = str(ViUtils.remove_accents(input_text_pre).decode('utf-8'))
    input_text_pre_accent = ViTokenizer.tokenize(input_text_pre)
    input_text_pre_no_accent = ViTokenizer.tokenize(input_text_pre_no_accent)
    input_pre = []
    input_pre.append(input_text_pre_accent)
    input_pre.append(input_text_pre_no_accent)

    print(input_pre)


def handle_negation(text):
    negation_words = ['không', 'chưa', 'chẳng', 'chả', 'hết', "tệ", "xấu"]
    positive_words = ['tốt', 'tuyệt', 'đẹp', 'hợp', 'ổn', 'xinh', 'ưng', 'ngon', 'có', 'thể', 'nên']
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


def vi_en(text_vi,translator):
    # translater = EasyGoogleTranslate(source_language='vi', target_language='en')
    # return translater.translate(text_vi)
    # translator = Translate()
    translated = translator.translate('en','vi',text_vi)
    return translated['Result']


def en_vi(text_en,translator):
    # translator = Translate()
    translated = translator.translate('vi', 'en', text_en)
    return translated['Result']


async def pre_data(start, end, input_data, label_data, model, tokenizer):
    res = []
    res_label = []
    translator = Translate()
    f = open('pre_data.txt', 'a', encoding='utf-8')
    write_label = open('pre_label.txt', 'a', encoding='utf-8')
    while start <= end:
        print(start)
        print(input_data[start])
        input_text_pre = clean_emoji(input_data[start])  # xóa biểu tượng cảm xúc
        input_text_pre = preprocessingData(input_text_pre)  # chuẩn hóa cơ bản (xóa dấu câu, kí tự đặc biệt,...)
        input_text_pre = clean_duplicate(input_text_pre)  # xóa kí tự trùng lặp
        input_text_pre = clean_acronyms(input_text_pre)  # chuẩn hóa từ viết tắt

        vi_en_text = vi_en(input_text_pre, translator)  # chuyển sang tiếng anh
        en_vi_text = en_vi(vi_en_text, translator)  # chuyển về sang tiếng Việt

        en_vi_text_accent = tokenizer_word(en_vi_text)  # phân tách từ
        en_vi_text_accent = clean_stopword(en_vi_text_accent)  # xóa stopp words
        en_vi_text_accent = pos(en_vi_text_accent, model=model, tokenizer=tokenizer)  # pos tagging
        en_vi_text_accent = handle_negation(en_vi_text_accent)  # xử lí phủ định

        en_vi_text_no_accent = str(ViUtils.remove_accents(en_vi_text_accent).decode('utf-8'))  # xóa dấu từ

        input_text_pre_accent = tokenizer_word(input_text_pre)  # phân tách từ
        input_text_pre_accent = clean_stopword(input_text_pre_accent)  # xóa stopp words
        input_text_pre_accent = pos(input_text_pre_accent, model=model, tokenizer=tokenizer)  # pos tagging

        input_text_pre_accent = handle_negation(input_text_pre_accent)  # xử lí phủ định

        input_text_pre_no_accent = str(ViUtils.remove_accents(input_text_pre_accent).decode('utf-8'))  # xóa dấu từ

        res.append(en_vi_text_accent)
        res.append(en_vi_text_no_accent)
        res.append(input_text_pre_accent)
        res.append(input_text_pre_no_accent)

        # ghi dữ liệu vào file - để tái sử dụng khi train mô hình nhanh hơn
        f.write(en_vi_text_accent)
        f.write("\n")
        f.write(en_vi_text_no_accent)
        f.write("\n")
        f.write(input_text_pre_accent)
        f.write("\n")
        f.write(input_text_pre_no_accent)
        f.write("\n")

        res_label.append(label_data[start])
        res_label.append(label_data[start])
        res_label.append(label_data[start])
        res_label.append(label_data[start])

        write_label.write(label_data[start])
        write_label.write('\n')
        write_label.write(label_data[start])
        write_label.write('\n')
        write_label.write(label_data[start])
        write_label.write('\n')
        write_label.write(label_data[start])
        write_label.write('\n')

        start += 1

    f.close()
    write_label.close()

    return res, res_label


async def load_data():
    tokenizer, model = load_mlm_model()
    data = pd.read_csv("../data/data - data.csv")
    # print(data.head())
    sentiment_data = pd.DataFrame({'sentence': data['comment'], 'label': data['label']})
    sentiment_data = sentiment_data.dropna()
    sentiment_data = sentiment_data.reset_index(drop=True)
    # print(sentiment_data.head())
    input_data = sentiment_data['sentence'].values
    input_label = sentiment_data['label'].values
    input_pre = []
    label_with_accent = []
    # for i, data in enumerate(input_data):
    #     input_text_pre = preprocessingData(data)  # chuẩn hóa cơ bản (xóa dấu câu, kí tự đặc biệt,...)
    #     input_text_pre = clean_emoji(input_text_pre)  # xóa biểu tượng cảm xúc
    #     input_text_pre = clean_duplicate(input_text_pre)  # xóa kí tự trùng lặp
    #     input_text_pre = clean_acronyms(input_text_pre)  # chuẩn hóa từ viết tắt
    #     # vi_en_text = vi_en(input_text_pre)  # chuyển sang tiếng anh
    #     # en_vi_text = en_vi(vi_en_text)  # chuyển về sang tiếng Việt
    #     # en_vi_text_accent = tokenizer_word(en_vi_text)  # phân tách từ
    #     # en_vi_text_accent = clean_stopword(en_vi_text_accent)  # xóa stopp words
    #     # en_vi_text_accent = pos(en_vi_text_accent)  # pos tagging
    #     # en_vi_text_accent = handle_negation(en_vi_text_accent)  # xử lí phủ định
    #
    #     # en_vi_text_no_accent = str(ViUtils.remove_accents(en_vi_text_accent).decode('utf-8'))  # xóa dấu từ
    #     input_text_pre_accent = tokenizer_word(input_text_pre)  # phân tách từ
    #     input_text_pre_accent = clean_stopword(input_text_pre_accent)  # xóa stopp words
    #     input_text_pre_accent = pos(input_text_pre_accent, model=model, tokenizer=tokenizer)  # pos tagging
    #     print(str(i) + "-> " + input_text_pre_accent)
    #     input_text_pre_accent = handle_negation(input_text_pre_accent)  # xử lí phủ định
    #
    #     input_text_pre_no_accent = str(ViUtils.remove_accents(input_text_pre_accent).decode('utf-8'))  # xóa dấu từ
    #
    #     input_pre.append(input_text_pre_accent)
    #     input_pre.append(input_text_pre_no_accent)
    #     # input_pre.append(en_vi_text_accent)
    #     # input_pre.append(en_vi_text_no_accent)
    #
    #     label_with_accent.append(input_label[i])

    # 0-1000
    # res_pre_0_1000, res_label_0_1000 = await pre_data(0, 1000, input_data=input_data, label_data=input_label, model=model,
    #                                           tokenizer=tokenizer)
    # input_pre += res_pre_0_1000
    # label_with_accent += res_label_0_1000
    # print(input_pre)

    # 1001-2000
    # res_pre_1001_2000, res_label_1001_2000 = await pre_data(1001, 2000, input_data=input_data, label_data=input_label, model=model,
    #                                         tokenizer=tokenizer)
    # input_pre += res_pre_1001_2000
    # label_with_accent += res_label_1001_2000
    #
    # # 2001-3000
    # res_pre_2001_3000, res_label_2001_3000 = await pre_data(2001, 3000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_2001_3000
    # label_with_accent += res_label_2001_3000

    # 3001-4000
    # res_pre_3001_4000, res_label_3001_4000 = await pre_data(3001, 4000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_3001_4000
    # label_with_accent += res_label_3001_4000

    # # 4001-5000
    # res_pre_4001_5000, res_label_4001_5000 = await pre_data(4001, 5000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_4001_5000
    # label_with_accent += res_label_4001_5000
    #
    # # 5001-6000
    # res_pre_5001_6000, res_label_5001_6000 = await pre_data(5001, 6000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_5001_6000
    # label_with_accent += res_label_5001_6000

    # 6001-7000
    # res_pre_6001_7000, res_label_6001_7000 = await pre_data(6001, 7000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_6001_7000
    # label_with_accent += res_label_6001_7000
    #
    # # 7001-8000
    # res_pre_7001_8000, res_label_7001_8000 = await pre_data(7001, 8000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_7001_8000
    # label_with_accent += res_label_7001_8000
    #
    # # 8001-9000
    # res_pre_8001_9000, res_label_8001_9000 = await pre_data(8001, 9000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_8001_9000
    # label_with_accent += res_label_8001_9000
    #
    # 9001-10000
    # res_pre_9001_10000, res_label_9001_10000 = await pre_data(9001, 10000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_9001_10000
    # label_with_accent += res_label_9001_10000
    #
    # # 10001-11000
    # res_pre_10001_11000, res_label_10001_11000 = await pre_data(10001, 11000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_10001_11000
    # label_with_accent += res_label_10001_11000
    #
    # # 11001-12000
    # res_pre_11001_12000, res_label_11001_12000 = await pre_data(11001, 12000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_11001_12000
    # label_with_accent += res_label_11001_12000

    # 12001-13000
    # res_pre_12001_13000, res_label_12001_13000 = await pre_data(12001, 13000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_12001_13000
    # label_with_accent += res_label_12001_13000
    #
    # # 13001-14000
    # res_pre_13001_14000, res_label_13001_14000 = await pre_data(13001, 14000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_13001_14000
    # label_with_accent += res_label_13001_14000
    #
    # # 14001-15000
    # res_pre_14001_15000, res_label_14001_15000 = await pre_data(14001, 15000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_14001_15000
    # label_with_accent += res_label_14001_15000

    # 15001-16000
    # res_pre_15001_16000, res_label_15001_16000 = await pre_data(15001, 16000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_15001_16000
    # label_with_accent += res_label_15001_16000
    #
    # # 16001-17000
    # res_pre_16001_17000, res_label_16001_17000 = await pre_data(16001, 17000, input_data=input_data, label_data=input_label,
    #                                                   model=model,
    #                                                   tokenizer=tokenizer)
    # input_pre += res_pre_16001_17000
    # label_with_accent += res_label_16001_17000
    #
    # # 17001-18000
    # res_pre_17001_18000, res_label_17001_18000 = await pre_data(17001, 18000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_17001_18000
    # label_with_accent += res_label_17001_18000


    # 18001-19000
    # res_pre_18001_19000, res_label_18001_19000 = await pre_data(18001, 19000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_18001_19000
    # label_with_accent += res_label_18001_19000
    #
    # # 19001-20000
    # res_pre_19001_20000, res_label_19001_20000 = await pre_data(19001, 20000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_19001_20000
    # label_with_accent += res_label_19001_20000
    #
    # # 20001-21000
    # res_pre_20001_21000, res_label_20001_21000 = await pre_data(20001, 21000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_20001_21000
    # label_with_accent += res_label_20001_21000

    # # 21001-22000
    # res_pre_21001_22000, res_label_21001_22000 = await pre_data(21001, 22000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_21001_22000
    # label_with_accent += res_label_21001_22000
    #
    # # 22001-23000
    # res_pre_22001_23000, res_label_22001_23000 = await pre_data(22001, 23000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_22001_23000
    # label_with_accent += res_label_22001_23000
    #
    # # 23001-24000
    # res_pre_23001_24000, res_label_23001_24000 = await pre_data(23001, 24000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_23001_24000
    # label_with_accent += res_label_23001_24000

    # 24001-25000
    # res_pre_24001_25000, res_label_24001_25000 = await pre_data(24001, 25000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_24001_25000
    # label_with_accent += res_label_24001_25000
    #
    # # 25001-26000
    # res_pre_25001_26000, res_label_25001_26000 = await pre_data(25001, 26000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_25001_26000
    # label_with_accent += res_label_25001_26000
    #
    # # 26001-27000
    # res_pre_26001_27000, res_label_26001_27000 = await pre_data(26001, 27000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_26001_27000
    # label_with_accent += res_label_26001_27000

    # 27001-28000
    # res_pre_27001_28000, res_label_27001_28000 = await pre_data(27001, 28000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_27001_28000
    # label_with_accent += res_label_27001_28000
    #
    # # 28001-29000
    # res_pre_28001_29000, res_label_28001_29000 = await pre_data(28001, 29000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_28001_29000
    # label_with_accent += res_label_28001_29000
    #
    # # 29001-30000
    # res_pre_29001_30000, res_label_29001_30000 = await pre_data(29001, 30000, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_29001_30000
    # label_with_accent += res_label_29001_30000
    #
    # # 30001-31000
    # res_pre_30001_31459, res_label_30001_31459 = await pre_data(30001, 31459, input_data=input_data, label_data=input_label,
    #                                                       model=model,
    #                                                       tokenizer=tokenizer)
    # input_pre += res_pre_30001_31459
    # label_with_accent += res_label_30001_31459


def cal_avg_length():
    input_pre = []
    with open('pre_data.txt', encoding='utf-8', mode='r') as f:
        line = [s.strip() for s in f.readlines()]
        input_pre = line

    f.close()

    seq_len = [len(i.split()) for i in input_pre[0:5000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2500

    seq_len = [len(i.split()) for i in input_pre[5000:10000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2700

    seq_len = [len(i.split()) for i in input_pre[10000:15000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2000

    seq_len = [len(i.split()) for i in input_pre[15000:20000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2000

    seq_len = [len(i.split()) for i in input_pre[20000:25000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 4300

    seq_len = [len(i.split()) for i in input_pre[25000:30000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 3000

    seq_len = [len(i.split()) for i in input_pre[30000:35000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2000

    seq_len = [len(i.split()) for i in input_pre[35000:40000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2500

    seq_len = [len(i.split()) for i in input_pre[40000:45000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2500

    seq_len = [len(i.split()) for i in input_pre[45000:50000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2600

    seq_len = [len(i.split()) for i in input_pre[50000:55000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2500

    seq_len = [len(i.split()) for i in input_pre[55000:60000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2700

    seq_len = [len(i.split()) for i in input_pre[60000:65000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2900

    seq_len = [len(i.split()) for i in input_pre[65000:70000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2500

    seq_len = [len(i.split()) for i in input_pre[70000:75000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 3000

    seq_len = [len(i.split()) for i in input_pre[75000:80000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2600

    seq_len = [len(i.split()) for i in input_pre[80000:85000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2900

    seq_len = [len(i.split()) for i in input_pre[85000:90000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2500

    seq_len = [len(i.split()) for i in input_pre[90000:95000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 3200

    seq_len = [len(i.split()) for i in input_pre[95000:100000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 3000

    seq_len = [len(i.split()) for i in input_pre[100000:105000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 1900

    seq_len = [len(i.split()) for i in input_pre[105000:110000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 2600

    seq_len = [len(i.split()) for i in input_pre[110000:115000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 1900

    seq_len = [len(i.split()) for i in input_pre[115000:120000]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # => 3000

    seq_len = [len(i.split()) for i in input_pre[120000:125760]]
    pd.Series(seq_len).hist(bins=10)
    plt.show() # =>3300

    return input_pre

def build_vocabulary():
    input_pre = []
    label_with_accent_data = []
    label_dict = {'NEG': 0, 'NEU': 1, 'POS': 2}
    with open('pre_data.txt', encoding='utf-8', mode='r') as f:
        input_pre = [s.strip() for s in f.readlines()]

    with open('pre_label.txt', encoding='utf-8') as fl:
        label_with_accent_data = [s.strip() for s in fl.readlines()]

    fl.close()
    f.close()

    label_index = [label_dict[i] for i in label_with_accent_data]
    label_tf = tf.keras.utils.to_categorical(label_index, num_classes=3, dtype='float32')
    tokenizer_data = Tokenizer(oov_token='<OOV>', filters= '', split= ' ')
    tokenizer_data.fit_on_texts(input_pre)

    tokenizer_data_text = tokenizer_data.texts_to_sequences(input_pre)
    vec_data = pad_sequences(tokenizer_data_text, padding='post',maxlen= 2500)

    pickle.dump(tokenizer_data, open("tokenizer_data.plk", "wb"))

    print('input data.shape:', vec_data.shape)
    data_voc_size = len(tokenizer_data.word_index)+1
    print('data voc size ', data_voc_size)

    x_train, x_val, y_train, y_val = train_test_split(vec_data, label_tf, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size=0.1, random_state=42)
    print('training sample: ', len(x_train))
    print('validation sample: ',len(x_val))
    print('test sample: ',len(x_test))

    return data_voc_size,x_val,y_val,x_train,y_train,x_test,y_test


def build_sentiment_model():

    dropout_threshold = 0.4
    input_dim = build_vocabulary()[0]
    output_dim = 32
    input_length = 2500

    initializer = initializers.GlorotNormal()

    input_layer = Input(shape=input_length)
    feature = Embedding(input_dim= input_dim, output_dim= output_dim, input_length= input_length, embeddings_initializer="GlorotNormal")(input_layer)

    cnn_feature = Conv1D(filters= 32, kernel_size= 3, padding= 'same', activation= 'relu')(feature)
    cnn_feature = MaxPooling1D()(cnn_feature)
    cnn_feature = Dropout(dropout_threshold)(cnn_feature)
    cnn_feature = Conv1D(filters= 32, kernel_size= 3, padding= 'same', activation= 'relu')(cnn_feature)
    cnn_feature = MaxPooling1D()(cnn_feature)
    cnn_feature = LayerNormalization()(cnn_feature)
    cnn_feature = Dropout(dropout_threshold)(cnn_feature)

    bi_lstm_feature = Bidirectional(
        LSTM(units=32, dropout= dropout_threshold, return_sequences= True, kernel_initializer= initializer),
        merge_mode= 'concat') (feature)
    # bi_lstm_feature = Attention()([bi_lstm_feature,bi_lstm_feature])
    bi_lstm_feature = MaxPooling1D()(bi_lstm_feature)

    bi_lstm_feature = Bidirectional(
        GRU(units=32, dropout=dropout_threshold, return_sequences=True, kernel_initializer=initializer),
        merge_mode='concat')(bi_lstm_feature)
    bi_lstm_feature = MaxPooling1D()(bi_lstm_feature)
    bi_lstm_feature = LayerNormalization()(bi_lstm_feature)

    combine_feature = Concatenate()([cnn_feature, bi_lstm_feature])
    combine_feature = GlobalMaxPooling1D()(combine_feature)
    combine_feature = LayerNormalization()(combine_feature)

    classifier = Dense(90, activation= 'relu')(combine_feature)
    classifier = Dropout(0.2)(classifier)
    classifier = Dense(70,activation= 'relu')(classifier)
    classifier = Dropout(0.2)(classifier)
    classifier = Dense(50, activation='relu')(classifier)
    classifier = Dropout(0.2)(classifier)
    classifier = Dense(30, activation='relu')(classifier)
    classifier = Dropout(0.2)(classifier)
    classifier = Dense(3, activation='softmax')(classifier)

    model = Model(inputs = input_layer, outputs = classifier)
    return model


def train_model():
    data_voc_size, x_val, y_val, x_train, y_train, x_test, y_test = build_vocabulary()
    model = build_sentiment_model()
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    calback_model = callbacks.ModelCheckpoint('sentiment_model_bi-lstm_cnn.h5', monitor='val-loss')
    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=20, batch_size=128,
                        callbacks=[calback_model])
    #
    # load weight model
    model.load_weights('sentiment_model_bi-lstm_cnn.h5')
    model.evaluate(x_test,y_test)


async def pre_raw_input(raw_input, tokenizer):
    tokenizer_MLM, model_MLM = await load_mlm_model()

    input_text_pre = preprocessingData(raw_input)  # chuẩn hóa cơ bản (xóa dấu câu, kí tự đặc biệt,...)
    input_text_pre = clean_emoji(input_text_pre)  # xóa biểu tượng cảm xúc
    input_text_pre = clean_duplicate(input_text_pre)  # xóa kí tự trùng lặp
    input_text_pre = clean_acronyms(input_text_pre)  # chuẩn hóa từ viết tắt
    input_text_pre_accent = tokenizer_word(input_text_pre)  # phân tách từ
    input_text_pre_accent = clean_stopword(input_text_pre_accent)  # xóa stopp words
    input_text_pre_accent = pos(input_text_pre_accent, model=model_MLM, tokenizer=tokenizer_MLM)  # pos tagging

    input_text_pre_accent = handle_negation(input_text_pre_accent)  # xử lí phủ định

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
    vec_data = await pre_raw_input(raw_input,tokenizer)
    result,conf = await inference_model(vec_data,model)

    return result,conf


async def load_sentiment_model():
    return load_model('sentiment_model_bi-lstm_cnn.h5')


async def load_voc():
    with open('tokenizer_data.plk', 'rb') as input_file:
        my_tokenizer = pickle.load(input_file)

    return my_tokenizer


def show_model_architecture(model):
    image_file = 'model_architecture.png'
    plot_model(model = model, to_file= image_file, show_shapes=True, dpi= 300)


if __name__ == '__main__':
    # tokenizer, model = load_mlm_model()
    # input = r"kh thì Nchung a 😅 👠 😆 lô ối giời là @ 23123 về (rấttttt ổn ❤️. Khá đc"
    # input = r"Nam tham gia vào các hoạt động học tập, không vi phạm quy chế."
    # input = r"Xe mới khá đẹp , chất lượng đi tour thì kh có gì phải bàn ........"
    # input = "Sản phẩm không tệ, cũng tạm được."
    #
    # # pr(input)
    # print(pos_tagging(input))
    # input = preprocessingData(input)
    # input = clean_emoji(input)
    # input = clean_acronyms(input)
    # input = clean_duplicate(input)
    #
    # input = clean_stopword(input)
    # input = pos(input, model=model, tokenizer=tokenizer)
    # input = tokenizer_word(input)
    # # print(input)
    # input = handle_negation(input)
    # print(input)
    # input = tokenizer_word(input)
    # pos_tagging(input)
    # print(pos_tag(input))
    # # print(input)
    # input = clean_stopword(input)

    # print(input)
    # print(mlm({"inputs": input}))

    # str1 = r"quuuá"
    # print(''.join(dict.fromkeys(str1))) # xoá từ trùng lặp

    # pr(input)
    # s = r'Ðảm baỏ chất lựơng phòng thí nghịêm hoá học'
    # s = text_normalize(s)
    # print(s)

    s = r"Nam tham gia vào các hoạt động học tập, không vi phạm quy chế."
    s1 = "Nam không tham gia vào các hoạt động học tập, vi phạm quy chế."
    s2 = "Không, bạn có thích học"
    # print(vi_en(s))
    # print(tf.__version__)
    text_masked = "Ngày mai <mask> tôi <mask> học. Phải học <mask> chăm chỉ."
    # print(count_mask(text_masked))
    # print(mlm(text_masked,model=model,tokenizer=tokenizer))
    # print(full_mlm(text_masked,model=model,tokenizer=tokenizer))
    # asyncio.run(load_data())
    # cal_avg_length()
    # build_vocabulary()
    # train_model()
    moder_dir = 'sentiment_model_bi-lstm_cnn.h5'
    my_model = load_model(moder_dir)
    with open('tokenizer_data.plk', 'rb') as input_file:
        my_tokenizer = pickle.load(input_file)

    sentences = ["hơi nhỏ vải mỏng , mặc cx đc hình ảnh mang to hs chất lấy xu",
                 "Sản phẩm đẹp vải một lớp có hơi mỏng nhưng rất đẹp lần sau sẽ ủng hộ shop tiếp",
                 "Áo siêu siêu xinh lun mn ơi, mình mua 2áo thật đáng tiền áo xinh xĩu vải tốt viền 2 lớp giao hàng nhanh đóng gói kĩ càng ít chỉ thừa áo tôn dáng. Lúc đầu tưởng ko xinh nhưng khi nhận hàng r thì xinh ko tưởng mn ạ. Mn nên muaa về mặc bảo đảm sẽ mê. Áo đẹp mà giá lại hời nữa. Nhất định sẽ ủng hộ shoppp nhìu hơn nữaaa",
                 "Ở trên ảnh thì mình thấy là vải khác còn ở ngoài thì vải khác á mng=)). Nma kh s hết í mng mình có hơi thất vọng về vải ở trên mạng và ngoài đơi thôi còn mặc lên thì xinh lắm nhe <3. Nchung thì mặt đằng sau của áo thì mình ưng nhất",
                 "Huhu mình thấy áo ở ngoài với trong nó hơi kì chắc do mặc ko hợp",
                 "vải mỏng nhé nhìn cũng đẹp đẹp nói chung 55k vậy là tốt rồi kh mong đợi nhiều",
                 "Sản phẩm hơi thô, chất lượng tạm được, giao hàng nhanh,ok ok.",
                 "K hợp giá tiền k nên mua",
                 "Tệ sản phầm giao nhanh nút rớt. Bung chỉ có nhiều chỉ thừa. Ko nên mua. Không có ảnh mình lấy ảnh này cân nhắc trc khi mua",
                 "Áo nhận về bị rách nữa . Với giá này thì nghĩ xem mong chờ dc gì chẳnb qua thấy dth thì mua chơi thôi săn slae lần đầu trong đời mà bị v :)))))))))"]

    print(asyncio.run(prediction(r"Pô ok nghe nổ lực nhưng sơn đen bị dính lên miếng ixon pô giá rẻ.",my_tokenizer, my_model)))
    # for cS in sentences:
    #     print(cS)
    #     print(asyncio.run(
    #         prediction(cS, my_tokenizer, my_model)))

    # show_model_architecture(model= my_model)

    # print(sklearn.__version__)


