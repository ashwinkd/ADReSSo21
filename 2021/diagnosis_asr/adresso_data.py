from _global_vars import *


class ADReSSoData:
    def __init__(self, data_directory=DATA_DIR):
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
        for category in ['ad', 'cn']:
            category_path = os.path.join(segmentation_path, category)
            for speaker in os.listdir(category_path):
                spkr_path = os.path.join(category_path, speaker)
                spkr_segmentation = pd.read_csv(spkr_path)
                segments = self.get_participant_data(spkr_segmentation)
                segmentation_data[speaker.split('.')[0]] = segments
        return segmentation_data

    def get_score_data(self):
        score_path = os.path.join(self.data_directory, 'adresso-train-mmse-scores.csv')
        score_df = pd.read_csv(score_path)
        score_df.rename(columns={"adressfname": "speaker"}, inplace=True)
        return score_df

    def get_audio_paths(self):
        audio_data = {}
        audio_path = os.path.join(self.data_directory, 'wavefiles')
        for category in ['ad', 'cn']:
            category_path = os.path.join(audio_path, category)
            for speaker in os.listdir(category_path):
                adressfname = speaker.split('.')[0]
                audio_data[adressfname] = os.path.join(category_path, speaker)
        return audio_data

    def get_data(self):
        speaker_file = 'speaker_data_segments.csv'
        # speaker_data = pd.read_csv('speaker_data_silence_old.csv')
        speaker_data = self.get_score_data()
        speaker_data['segments'] = speaker_data['speaker'].map(self.get_segmentation_data())
        # speaker_data['audio'] = speaker_data['speaker'].map((self.get_audio_paths()))
        # transcript_data, phoneme_data, phoneme_rate_data = self.get_transcripts(speaker_data)
        # speaker_data['transcript'] = speaker_data['speaker'].map((transcript_data))
        # speaker_data['phonemes'] = speaker_data['speaker'].map((phoneme_data))
        # speaker_data['phoneme_rate'] = speaker_data['speaker'].map((phoneme_rate_data))
        # # speaker_data['silences'] = speaker_data['speaker'].map((self.get_silence(speaker_data)))
        # # speaker_data['n_short_long_pause'] = speaker_data['silences'].apply(
        # #     lambda x: self.get_pauses(vad_arrays=literal_eval(x),
        # #                               short_pause_length=150,
        # #                               long_pause_length=1000))
        # speaker_data[['speaker', 'mmse', 'dx', 'transcript', 'phonemes', 'phoneme_rate']].to_csv(speaker_file)
        speaker_data[['speaker', 'segments']].to_csv(speaker_file)
        return speaker_data

    def clean_up(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def get_transcripts(self, speaker_data):
        transcript_data = {}
        phoneme_data = {}
        phoneme_rate_data = {}
        for index, row in speaker_data[['speaker', 'segments', 'audio']].iterrows():
            speaker, segments, audio = row
            transcript = []
            phonemes = []
            phoneme_count = 0
            spkr_duration = 0
            for fname in self.segment_wav(speaker, segments, audio):
                try:
                    result = self.clean_up(self.transcribe_file(fname))
                    duration = self.get_duration(fname)
                    ph = self.get_phonemes(result)
                    phoneme_count += len(ph.split())
                    spkr_duration += duration
                    transcript.append(result)
                    phonemes.append(ph)
                except:
                    pass
            transcript_data[speaker] = transcript
            phoneme_data[speaker] = phonemes
            phoneme_rate_data[speaker] = phoneme_count / spkr_duration if spkr_duration != 0 else 0
        return transcript_data, phoneme_data, phoneme_rate_data

    def transcribe_file(self, speech_file):
        full_result = ""
        """Transcribe the given audio file."""

        client = speech.SpeechClient()

        with io.open(speech_file, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                          sample_rate_hertz=16000,
                                          language_code="en-US",
                                          enable_automatic_punctuation=True,
                                          use_enhanced=True,
                                          model="phone_call",
                                          )

        response = client.recognize(config=config, audio=audio)

        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            full_result += result.alternatives[0].transcript
            # print(u"Transcript: {}".format(result.alternatives[0].transcript))
        return full_result

    def get_duration(self, fname):
        with contextlib.closing(wave.open(fname, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration

    def get_phonemes(self, text):
        phoneme_list = []
        for word in text.split():
            try:
                phoneme = pronouncing.phones_for_word(word)
                phoneme_list.append(phoneme[0])
            except:
                pass
        return " ".join(phoneme_list)

    def segment_wav(self, speaker, segments, audio):
        segmented_audio = []
        audio_file = AudioSegment.from_wav(audio)
        info = mediainfo(audio)
        audio_file = audio_file.set_frame_rate(16000)
        for begin, end in segments:
            segment_path = 'tmp/{}_{}_{}.wav'.format(speaker, begin, end)
            audio_segment = audio_file[begin:end]
            audio_segment.export(segment_path, format="wav")
            # command = "ffmpeg -y -i {} -acodec pcm_s16le -b:a 256k -ac 1 -ar 16000 {}".format(segment_path,
            #                                                                                   segment_path)
            # os.system(command)
            segmented_audio.append(segment_path)
        return segmented_audio

    def compute_vad(self, filename):
        data = bob.io.audio.reader(filename)
        fl = (data.duration * 1000) / data.number_of_samples
        print(fl)
        DNN_VAD_labels = bob.kaldi.compute_dnn_vad(data.load()[0], data.rate)
        # return DNN_VAD_labels

    def get_silence(self, speaker_data):
        vad_data = {}
        for index, row in speaker_data[['speaker', 'segments', 'audio']].iterrows():
            speaker, segments, audio = row
            vad = []
            for filename in self.segment_wav(speaker, segments, audio):
                try:
                    result = self.compute_vad(filename)
                    if result:
                        vad.append(result)
                except Exception as e:
                    print(filename, "\n", e)
            vad_data[speaker] = vad
        return vad_data

    def remove_end_zeros(self, vad_array):
        vad_array = vad_array[vad_array.index(1):][::-1]
        vad_array = vad_array[vad_array.index(1):][::-1]
        return vad_array

    def get_pauses(self, vad_arrays, short_pause_length, long_pause_length, frame_length=25, hop_length=10):
        total_short_pause = 0
        total_long_pause = 0
        n_short = (short_pause_length - hop_length) / (frame_length - hop_length)
        n_long = (long_pause_length - hop_length) / (frame_length - hop_length)
        for vad_array in vad_arrays:
            try:
                vad_array = self.remove_end_zeros(vad_array)
                zero_run_lengths = self.get_zero_run_lengths(vad_array)
                num_short_pause = self.get_pause_count(run_lengths=zero_run_lengths, n_min=n_short, n_max=n_long)
                num_long_pause = self.get_pause_count(run_lengths=zero_run_lengths, n_min=n_long)
                total_short_pause += num_short_pause
                total_long_pause += num_long_pause
            except Exception as e:
                print(e)
                return
        return total_short_pause, total_long_pause

    def zero_runs(self, vad_array):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(vad_array, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    def get_zero_run_lengths(self, vad_array):
        zero_runs = self.zero_runs(vad_array)
        zero_run_lengths = np.diff(zero_runs, axis=1).T[0]
        return zero_run_lengths

    def get_pause_count(self, run_lengths, n_min, n_max=sys.maxsize):
        run_lengths = [int(elem / n_min) for elem in run_lengths if n_max > elem >= n_min]
        return sum(run_lengths)
