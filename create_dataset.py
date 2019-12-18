import os
import shutil
import zipfile
import urllib.request
import scipy.io
import h5py
import numpy as np

from constants import DATASET_FN
from constants import DATASET_URL
from constants import DATASET_HDF
from constants import DATA_PATH


def download_dataset():
    file_name = DATASET_FN
    u = urllib.request.urlopen(DATASET_URL)

    with open(file_name, 'wb') as f:
        meta = u.info()
        file_size = int(meta['Content-Length'])
        print("Downloading: {0:} {1:5.2f} MB".format(os.path.basename(file_name), file_size / (1024**2)))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d MB [%3.2f%%]" % (file_size_dl / (1024**2), file_size_dl * 100. / file_size)
            status = status + chr(8) * (len(status) + 1)
            print(status, end='')
    print('Download done!')


def extract():
    print('Extracting Files...')
    with zipfile.ZipFile(DATASET_FN) as zip:
        zip.extractall(DATA_PATH)


def convert():
    print('Converting to NumPy...')
    with h5py.File(DATASET_HDF, 'w') as hdf:
        for dirpath, dnames, fnames in os.walk(DATA_PATH):
            for f in fnames:
                if f.endswith(".mat"):
                    fn = os.path.join(dirpath, f)
                    split_path = fn.split(os.sep)

                    if 'volumetric_data' not in split_path or f == 'train_feature.mat' or f == 'test_feature.mat':
                        continue

                    category = split_path[-4]
                    arr = scipy.io.loadmat(fn)['instance'].astype(np.uint8)
                    arrpad = np.zeros((32,) * 3, dtype=np.uint8)
                    arrpad[1:-1, 1:-1, 1:-1] = arr

                    if category not in hdf:
                        hdf.create_group(category)

                    hdf[category].create_dataset(os.path.splitext(f)[0], data=arrpad, compression='gzip')


def cleanup():
    print('Cleaning up...')
    dataset_file = os.path.abspath(DATASET_FN)
    dataset_path = os.path.abspath(dataset_file)[:-8]
    if os.path.exists(dataset_file):
        os.remove(dataset_file)
    shutil.rmtree(dataset_path, ignore_errors=True)


if __name__ == '__main__':
    if os.path.exists(DATASET_HDF):
        print('HDF File exists, no need to re-download data')
    else:
        download_dataset()
        extract()
        convert()
        cleanup()
        print('Done!')
