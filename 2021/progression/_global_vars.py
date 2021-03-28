import os
import pandas as pd
from pydub import AudioSegment
from ast import literal_eval
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.model_selection import train_test_split
import random

random.seed(42)

DATA_DIR = '/home/ashwin/Documents/Documents/Work/Research/ADReSSo/ADReSSo21/progression/train/'
