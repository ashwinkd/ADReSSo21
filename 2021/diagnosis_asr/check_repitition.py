import re
from ast import literal_eval
from itertools import groupby

import pandas as pd

df = pd.read_csv('./speaker_data_transcripts.csv', usecols=['speaker', 'segments', 'transcript'])


def is_repeated(arr):
    arr = groupby(arr)
    for c in [sum(1 for _ in group) for _, group in groupby(arr)]:
        if c > 1:
            return True
    return False


for idx, row in df.iterrows():
    speaker, segments, transcripts = row
    for tr in literal_eval(transcripts):
        tr = re.sub("[^a-z ]", "", tr.lower()).split()
        if is_repeated(tr):
            print(tr)
