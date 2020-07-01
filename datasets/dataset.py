import torch
import collections
import os, glob
import soundfile as sf
import librosa
from torchvision import transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed

LOGICAL_DATA_ROOT = '/data3/luchao/LA'
ASVFile = collections.namedtuple('ASVFile',
                                 ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])


def compute_spectrogram_phase(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    mag, phase = librosa.magphase(s)
    phase_angle = np.angle(phase)
    # group delay in http://www.asvspoof.org/asvspoof2015/NTU.pdf
    phase_diff = phase[1:,:] - phase[:-1,:]
    diff_angle = np.angle(phase_diff)
    gd = diff_angle + 2 * np.pi

    return gd
def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s) ** 2
    # melspect = librosa.feature.melspectrogram(S=a)
    feat = librosa.power_to_db(a)
    return feat


def get_cqt_spectrum(x):
    s = librosa.core.cqt(x, sr=16000)
    feat = np.abs(s) ** 2
    feat = librosa.power_to_db(feat)
    return feat


def get_mfcc(x):
    s = librosa.core.stft(x, n_fft=3200, win_length=3200, hop_length=1600)
    a = np.abs(s) ** 2
    melspect = librosa.feature.melspectrogram(S=a)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspect))
    return mfcc


def compute_mfcc_feats(x):
    mfcc = get_mfcc(x)
    delta = librosa.feature.delta(mfcc)
    feats = np.concatenate((mfcc, delta), axis=0)
    return feats


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len) + 1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """

    def __init__(self, data_root=LOGICAL_DATA_ROOT, transform=None, sample_size=None,data_model='train'):
        self.prefix = 'ASVspoof2019_LA'
        self.data_model = data_model
        self.sysid_dict = {
            '-': 0,  # bonafide speech  # ASVspoof2019_LA_cm_protocols
            'A01': 1,  # neural waveform model
            'A02': 2,  # vocoder
            'A03': 3,  # vocoder
            'A04': 4,  # waveform concatenation
            'A05': 5,  # vocoder
            'A06': 6,  # spectral filtering
            'A07': 7,  # vocoder+GAN
            'A08': 8,  # neural waveform
            'A09': 9,  # vocoder
            'A10': 10,  # neural waveform
            'A11': 11,  # griffin lim
            'A12': 12,  # neural waveform
            'A13': 13,  # waveform concatenation+waveform filtering
            'A14': 14,  # vocoder
            'A15': 15,  # neural waveform
            'A16': 16,  # waveform concatenation
            'A17': 17,  # waveform filtering
            'A18': 18,  # vocoder
            'A19': 19  # spectral filtering
        }
        self.featrue = 'spectrogram_phase'
        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}
        self.data_root = data_root
        # if self.featrue == 'spectrogram_phase':
        #     # self.feat = compute_spectrogram_phase(x=None)
        if self.data_model == 'train':
            self.dset_name = 'train'
            self.protocols_fname = 'train.trn'
        elif self.data_model == 'dev':
            self.protocols_fname = 'dev.trl'
            self.dset_name = 'dev'
        else:
            self.protocols_fname = 'eval.trl'
            self.dset_name = 'eval'
        self.protocols_dir = os.path.join(self.data_root,
                                          '{}_cm_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, self.dset_name), 'flac')
        self.protocols_fname = os.path.join(self.protocols_dir,
                                            'ASVspoof2019.LA.cm.{}.txt'.format(self.protocols_fname))
        self.transform = transforms.Compose([
            lambda x: pad(x),
            lambda x: librosa.util.normalize(x),
            lambda x: compute_spectrogram_phase(x),
            # lambda x: librosa.feature.chroma_cqt(x, sr=16000, n_chroma=20),
            lambda x: Tensor(x)
        ])
        self.cache_fname_new = os.path.dirname(self.files_dir) +'/'+self.featrue
        if not os.path.exists(self.cache_fname_new):
            os.makedirs(self.cache_fname_new)
        self.npy_files = glob.glob(os.path.join(self.cache_fname_new, '*.npy'))  # .clear()
        if len(self.npy_files) != len(self.parse_protocols_file(self.protocols_fname)):
            self.npy_files.clear()  # 清空
            # print(self.cache_fname)
            a = 0
            files_meta = self.parse_protocols_file(self.protocols_fname)
            for i, item in enumerate(files_meta):
                file_name = item.file_name
                file_save_path = os.path.join(self.cache_fname_new, '{}.npy'.format(file_name))
                data_x, data_y, data_sysid = self.read_file(item)
                data_x = self.transform(data_x)
                torch.save((data_x, data_y, data_sysid, item), file_save_path)
                a = i + 1
                print(a)
            # data = list(map(self.read_file, self.files_meta))

            # self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            # if self.transform:
            #     # self.data_x = list(map(self.transform, self.data_x)) 
            #     self.data_x = Parallel(n_jobs=1, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)
            # torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            print('Dataset saved to cache ', a)
            self.npy_files = glob.glob(os.path.join(self.cache_fname_new, '*.npy'))
        if sample_size:
            select_idx = np.random.choice(len(self.npy_files), size=(sample_size,), replace=True).astype(
                np.int32)  # 从数组、列表或元组中随机抽取
            self.npy_files = [self.npy_files[x] for x in select_idx]
        self.length = len(self.npy_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        npy = self.npy_files[idx]
        x, y, _, files_meta = torch.load(npy)
        return x, y, files_meta

    def read_file(self, meta):
        data_x, sample_read = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
                       file_name=tokens[1],
                       path=os.path.join(self.files_dir, tokens[1] + '.flac'),
                       sys_id=self.sysid_dict[tokens[3]],
                       key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)


if __name__ == '__main__':
    train_loader = ASVDataset(LOGICAL_DATA_ROOT, data_model='train')
    assert len(train_loader) == 25380, 'Incorrect size of training set.'
    dev_loader = ASVDataset(LOGICAL_DATA_ROOT,data_model='dev')
    assert len(dev_loader) == 24844, 'Incorrect size of dev set.'
    test_loader = ASVDataset(LOGICAL_DATA_ROOT, data_model='eval')
    print(len(test_loader))