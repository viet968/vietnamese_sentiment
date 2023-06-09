import requests
from bs4 import BeautifulSoup
import json

languages = ["1 Afrikaans af", "2 Albanian sq", "3 Arabic ar", "4 Armenian hy", "5 Azerbaijani az", "6 Basque eu",
             "7 Belarusian be", "9 Bulgarian bg", "9 Catalan ca", "10 Chinese (Simplified) zh-CN",
             "11 Chinese (Traditional) zh-TW", "12 Croatian hr", "13 Czech cs", "14 Danish da", "15 Dutch nl",
             "16 English en", "17 Estonian et", "18 Filipino tl", "19 Finnish fi", "20 French fr", "21 Galician gl",
             "22 Georgian ka", "23 German de", "24 Greek el", "25 Haitian Creole ht", "26 Hebrew iw", "27 Hindi hi",
             "28 Hungarian hu", "29 Icelandic is", "30 Indonesian id", "31 Irish ga", "32 Italian it",
             "33 Japanese ja", "34 Korean ko", "35 Latvian lv", "36 Lithuanian lt", "37 Macedonian mk", "38 Malay ms",
             "39 Maltese mt", "40 Norwegian no", "41 Persian fa", "42 Polish pl", "43 Portuguese pt",
             "44 Romanian ro", "45 Russian ru", "46 Serbian sr", "47 Slovak sk", "48 Slovenian sl", "49 Spanish es",
             "50 Swahili sw", "51 Swedish sv", "52 Thai th", "53 Turkish tr", "54 Ukrainian uk", "55 Urdu ur",
             "56 Vietnamese vi", "57 Welsh cy", "58 Yiddish yi"]


class Translate:
    def __init__(self):
        self

    def translate(self, to_language, from_language, thing):
        base_link = "http://translate.google.com/m?tl={}&sl={}&q={}"
        result = requests.get(base_link.format(to_language, from_language, thing)).content
        try:
            parse = BeautifulSoup(result, "html.parser")
            try:
                result = parse.find("div", {"class": "result-container"}).text
                show = {"From": from_language,
                        "To": to_language,
                        "Thing": thing,
                        "Result": result
                        }
                return show
            except:
                return "[ERROR] Error while translating"
        except:
            return "Something went wrong"

    def autodetect(self, to_language, thing):
        base_link = "http://translate.google.com/m?tl={}&sl=auto&q={}"
        result = requests.get(base_link.format(to_language, thing)).content
        try:
            parse = BeautifulSoup(result, "html.parser")
            try:
                result = parse.find("div", {"class": "result-container"}).text
                try:
                    from_language = parse.find("input", {"name": "hl"})["value"]
                except:
                    from_language = "Couldn't detect"
                show = {"From": from_language,
                        "To": to_language,
                        "Thing": thing,
                        "Result": result
                        }
                return show
            except:
                return "[ERROR] Error while translating"
        except:
            return "Something went wrong"

    def list_languages(self):
        return " " + str(languages).replace(",", "\n").rstrip()[1:-1:]