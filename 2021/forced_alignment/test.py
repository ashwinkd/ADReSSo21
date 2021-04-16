import json
import os

from pydub import AudioSegment
from pydub.playback import play

aligned_files = os.listdir(os.path.join('.', 'tmp', 'out'))
for file in os.listdir(os.path.join('.', 'tmp', 'audio')):
    audio = os.path.join('.', 'tmp', 'audio', file)
    segment_id = audio.split('\\')[-1].split('.')[0]
    aligned_filename = segment_id + ".json"
    if not aligned_filename in aligned_files:
        continue
    audio_file = AudioSegment.from_wav(audio)
    with open(os.path.join('.', 'tmp', 'out', aligned_filename)) as f:
        data = json.load(f)
    for word_fragment in data['fragments']:
        begin, children, end, wid, language, lines = word_fragment.values()
        begin = float(begin) * 1000
        end = float(end) * 1000
        word_segment = audio_file[begin:end]
        print(lines[0])
        play(word_segment)
        input()
