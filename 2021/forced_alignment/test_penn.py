from global_var import *

word_aligned_files = os.listdir(os.path.join('.', 'tmp', "penn_files", 'word_out'))
audio_files = os.listdir(os.path.join('.', 'tmp', 'audio'))

for file in r.sample(audio_files, len(audio_files)):
    audio = os.path.join('.', 'tmp', 'audio', file)
    segment_id = audio.split(directory_seperator)[-1].split('.')[0]
    aligned_filename = segment_id + ".pickle"
    if not aligned_filename in word_aligned_files:
        continue
    audio_file = AudioSegment.from_wav(audio)
    with open(os.path.join('.', 'tmp', "penn_files", 'word_out', aligned_filename), 'rb') as f:
        data = pickle.load(f)
    for (word, begin, end) in data:
        if word == "sp":
            continue
        begin = float(begin) * 1000
        end = float(end) * 1000
        word_segment = audio_file[begin:end]
        print(word)
        play(word_segment)
        input()
