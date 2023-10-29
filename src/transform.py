import librosa
import numpy as np

class MelTranform():
    """
    Tranformation class to get the MEL Spectrogram from the waveform
    """

    def __init__(self, n_mels=80, sample_rate=16000,
                 win_len=1024, hop_len=512, n_fft=None, to_db=True):
        """
        :param n_mels: number of MEL bands to create
        :param sample_rate: sample rate to deal with the waveforms
        :param win_len: window size of the FFT
        :param hop_len: step size of the FFT
        :param n_fft: number of FFT (default win_len)
        :param to_db: wether or not to convert to decibel
        """
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.win_len = win_len
        self.hop_len = hop_len
        self.to_db = to_db

        self.n_fft = n_fft if n_fft is not None else self.win_len

    def __call__(self, y, sample_rate=None):
        """
        Call method to convert a waveform to its MEL spectrogram representation

        :param y: waveform
        :param sample_rate: sample rate of the waveform if precised
        :return: MEL spectrogram of the waveform
        """

        # convert into float32 then change the sample rate if different from
        # the sample rate given for the class
        y = np.array(y).astype(np.float32)
        if sample_rate is not None and sample_rate != self.sample_rate:
            y = librosa.resample(y, sample_rate, self.sample_rate)

        # convert into a MEL Spectrogram
        spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
        )

        if self.to_db:
            spec = librosa.power_to_db(spec)

        return spec


class EmotionEncoder():
    """
    Tranformation class to encode and decode emotion id in array to a label
    """
    def __init__(self, labels=None, max_intensity=3):
        """
        :param labels: list of labels of the emotions
        :param max_intensity: maximum value an emotion can take
        """
        # set the labels if they are given else set it to default
        if labels is not None:
            self.str_to_id = {label: idx for idx, label in enumerate(labels)}
        else:
            self.str_to_id = {'angry':0, 'happy':1, 'neutral':2,'sad':3, 'other':4}
        self.id_to_str = {id:s for s, id in self.str_to_id.items()}

        self.max_intensity = max_intensity

    def __call__(self, emotion_str, intensity=None):
        """
        Call method to convert a given labeled emotion and intensity to
        a ponderated one_hot vector

        :param emotion_str: label representation of the emotion
        :param intensity: intensity of the emotion
        :return: Encoded emotion array
        """
        # convert the emotion into a one_hot encoding
        # set the unseen emotions into 'other'
        emotion = np.zeros(len(self.str_to_id))
        if emotion_str not in self.str_to_id:
            emotion_str = 'other'
        emotion[self.str_to_id[emotion_str]] = 1


        # ponderation of the one_hot vector given the max intensity
        if intensity is not None:
            emotion = emotion * intensity / self.max_intensity

        # special case to neutral emotion which have no intensity
        if emotion_str =='neutral':
            emotion[self.str_to_id[emotion_str]] = 2/3

        return emotion

    def encode(self, emotion, intensity=None):
        """
        Convert a given labeled emotion and intensity to a ponderated
        one_hot vector

        :param emotion_str: label representation of the emotion
        :param intensity: intensity of the emotion
        :return: Encoded emotion array
        """
        return self(emotion, intensity)

    def decode(self, emotion_id):
        """
        Convert a given id of emotion in the one_hot vector to
        the labeled emotion

        :param emotion_id: id representation of the emotion
        :return: label representation of the emotion
        """
        return self.id_to_str[np.argmax(emotion_id)]
