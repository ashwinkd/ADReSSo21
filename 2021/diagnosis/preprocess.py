from adresso_data import *
from SVR_model import *


def main():
    d = ADReSSoData()
    df_train, df_test = train_test_split(d.speaker_data, test_size=0.33, random_state=42)
    linguistic_model(df_train, df_test)
    print()


if __name__ == '__main__':
    main()
