import os
from ast import literal_eval

import pandas as pd

transcripts = pd.read_csv('./speaker_data_transcripts.csv', usecols=['speaker', 'segments', 'transcript'])

for idx, row in transcripts.iterrows():
    speaker, segments, transcript = row
    segments = literal_eval(segments)
    transcript = literal_eval(transcript)
    for segment, utterance in zip(segments, transcript):
        if utterance == "":
            source = os.path.join("tmp", "_".join([speaker] + [str(i) for i in list(segment)]) + ".wav")
            dest = os.path.join("blanks", "_".join([speaker] + [str(i) for i in list(segment)]) + ".wav")
            command = "cp {} {}".format(source, dest)
            os.system("cp {} {}".format(source, dest))
            print(command)
    print()
