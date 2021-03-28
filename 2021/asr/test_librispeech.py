#!/usr/bin/env python3
from jiwer import wer
import enchant
import os
import wave
from ast import literal_eval
from speechbrain.pretrained import TransformerASR

asr_model = TransformerASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                        savedir="pretrained_models/asr-transformer-transformerlm-librispeech")

# initialisation
d = enchant.Dict("en_US")


def get_vocab(vocabfile):
    vocab = ""
    text = open(vocabfile, 'r')
    for word in text.read().split('\n'):
        if d.check(word):
            vocab += word + " "
    return vocab.lower()


vocab = get_vocab('./vocab')

target_transcript = {f.split()[0]: " ".join(f.split()[1:]) for f in sorted(open('text_train').read().split('\n'))}
result_transcript = {}
audio_file = {f.split()[0]: " ".join(f.split()[1:]) for f in sorted(open('wav_train.scp').read().split('\n'))}
# model = Model("model")  # Uncomment to use Aspire Model
i = 0
avg_wer = 0
for key, file in audio_file.items():
    spkr = key.split('_')[0]
    if spkr not in result_transcript:
        result_transcript[spkr] = []
    try:
        result = asr_model.transcribe_file(file).lower()
        target = target_transcript[key]
        error = wer(target.split(), result.split())
        print("Target: ", target)
        print("Result: ", result)
        if result:
            result_transcript[spkr].append(result)
        print("WER: ", error)
        avg_wer += error
        i += 1
        print("Cumulative avg:", avg_wer / i)
        print()
    except Exception as e:
        print(e)

# output_fptr = open('output_text_test.csv', 'a')
# for spkr, utterances in result_transcript.items():
#     utterances = " . ".join(utterances)
#     line = "{},{}\n".format(spkr, utterances)
#     output_fptr.write(line)
# output_fptr.close()
