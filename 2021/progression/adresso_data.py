from _global_vars import *


class ADReSSoData:
    def __init__(self,
                 data_directory=DATA_DIR):
        self.data_directory = data_directory
        self.model = Model("model")
        self.speaker_data = self.get_data()

    def get_participant_data(self, df):
        participants_df = df.loc[df['speaker'] == 'PAR']
        begin = participants_df['begin'].tolist()
        end = participants_df['end'].tolist()
        return list(zip(begin, end))

    def get_segmentation_data(self):
        segmentation_data = {}
        segmentation_path = os.path.join(self.data_directory, 'segmentation')
        for category in ['decline', 'no_decline']:
            category_path = os.path.join(segmentation_path, category)
            for speaker in os.listdir(category_path):
                spkr_path = os.path.join(category_path, speaker)
                spkr_segmentation = pd.read_csv(spkr_path)
                segments = self.get_participant_data(spkr_segmentation)
                segmentation_data[speaker.split('.')[0]] = segments
        return pd.DataFrame.from_dict({'speaker': list(segmentation_data.keys()),
                                       'segments': list(segmentation_data.values())})

    def get_audio_paths(self):
        audio_data = {}
        audio_path = os.path.join(self.data_directory, 'wavefiles')
        for category in ['decline', 'no_decline']:
            category_path = os.path.join(audio_path, category)
            for speaker in os.listdir(category_path):
                adressfname = speaker.split('.')[0]
                audio_data[adressfname] = os.path.join(category_path, speaker)
        return audio_data

    def get_data(self):
        speaker_file = 'speaker_data.csv'
        if speaker_file not in os.listdir('..'):
            speaker_data = self.get_segmentation_data()
            # speaker_data['segments'] = speaker_data['speaker'].map()
            speaker_data['audio'] = speaker_data['speaker'].map((self.get_audio_paths()))
            speaker_data['utterances'] = speaker_data['speaker'].map((self.get_transcripts(speaker_data)))
            speaker_data.to_csv(speaker_file)
        else:
            speaker_data = pd.read_csv(speaker_file).dropna()
        return speaker_data

    def get_transcripts(self, speaker_data):
        transcript_data = {}
        for index, row in speaker_data[['speaker', 'segments', 'audio']].iterrows():
            speaker, segments, audio = row
            transcript = []
            for wf in self.segment_wav(speaker, segments, audio):
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                    print("Audio file must be WAV format mono PCM.")
                    exit(1)
                rec = KaldiRecognizer(self.model, wf.getframerate())
                result = self.asr(wf, rec)['text']
                if result:
                    transcript.append(result)
            transcript_data[speaker] = " . ".join(transcript)
        return transcript_data

    def asr(self, wf, rec):
        while True:
            data = wf.readframes(8000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = (rec.Result())
            else:
                res = (rec.PartialResult())
        return literal_eval(rec.FinalResult())

    def segment_wav(self, speaker, segments, audio):
        segmented_audio = []
        audio_file = AudioSegment.from_wav(audio)
        for begin, end in segments:
            segment_path = '/tmp/{}_{}_{}.wav'.format(speaker, begin, end)
            audio_segment = audio_file[begin:end]
            audio_segment.export(segment_path, format="wav")
            audio_segment = wave.open(segment_path, "rb")
            segmented_audio.append(audio_segment)
        return segmented_audio
