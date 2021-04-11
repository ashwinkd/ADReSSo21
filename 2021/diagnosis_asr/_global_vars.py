import os
import pandas as pd
from pydub import AudioSegment
from pydub.utils import mediainfo
from ast import literal_eval
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
# import bob.kaldi
# import bob.io.audio
import sys

import re
import pronouncing
import wave
import contextlib

from sklearn.model_selection import train_test_split
import random
from speechbrain.pretrained import TransformerASR
import io
import os

from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\adresso-asr-dab9362daad1.json"


random.seed(42)

DATA_DIR = 'F:\\Research\\ADReSSo\\ADReSSo21\\diagnosis\\train\\'
