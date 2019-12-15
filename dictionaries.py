import os
import numpy as np
from pathlib import Path

NRC_emotion_Lexicon_Wordlevelt_textfile = Path("./Data/dictionaries/NRC-Emotion-Lexicon-Wordlevel.txt")

NRC_pos_neg = dict()
NRC_lexicon = dict()

f = open(NRC_emotion_Lexicon_Wordlevelt_textfile, encoding="utf-8")
for line in f.readlines():
    word, category, value = line.split("\t")
    value = value.replace("\n", "")
    if(value == "1"):
        if(category == "positive" or category == "negative"):
            NRC_pos_neg[word] = category
        else:
            NRC_lexicon[word] = category
category_to_number_dict = {"Whataboutism,Straw_Men,Red_Herring": 0, "Loaded_Language":1, "Name_Calling,Labeling":2, "Flag-Waving":3,
"Exaggeration,Minimisation":4, "Causal_Oversimplification":5, "Repetition":6, "Slogans":7, "Black-and-White_Fallacy":8,
"Appeal_to_Authority":9, "Appeal_to_fear-prejudice":10, "Doubt":11, "Bandwagon,Reductio_ad_hitlerum":12,
"Thought-terminating_Cliches":13 }

number_to_category_dict = {0:"Whataboutism,Straw_Men,Red_Herring", 1:"Loaded_Language", 2:"Name_Calling,Labeling", 3:"Flag-Waving",
4:"Exaggeration,Minimisation", 5:"Causal_Oversimplification", 6:"Repetition", 7:"Slogans", 8:"Black-and-White_Fallacy",
9:"Appeal_to_Authority", 10:"Appeal_to_fear-prejudice", 11:"Doubt", 12:"Bandwagon,Reductio_ad_hitlerum",
13:"Thought-terminating_Cliches"}

if __name__ == "__main__":
    print(NRC_lexicon.keys())
    print(NRC_pos_neg.values())
