from functools import reduce

import pandas as pd

segments = pd.read_csv('./speaker_data_segments.csv', usecols=['speaker', 'segments'])
data = pd.read_csv('./speaker_data_phoneme_rate.csv')

df = reduce(lambda left, right: pd.merge(left, right, on='speaker'), [segments,
                                                                      data
                                                                      ])
df.to_csv('speaker_data_transcripts.csv', index=False)
