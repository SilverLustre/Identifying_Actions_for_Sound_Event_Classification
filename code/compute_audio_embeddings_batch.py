'''
__author__     = 'Benjamin Elizalde'
__email__      = 'bmartin1@alumni.cmu.edu'
__date__       = '2020'

'''

import pandas as pd
import numpy as np
import soundfile as sf
import openl3
import os


def compute_audio_embeddings(audio_path, metadata, batch_size=2000):
    metadata_file = pd.read_csv(metadata)
    filenames = metadata_file['filename'].tolist()

    filenames_num = len(filenames)
    emb_list = []

    # Best performing parameters
    content_type = "music"
    input_repr = "mel256"
    embedding_size = 6144
    model = openl3.models.load_audio_embedding_model(content_type=content_type,
                                                     input_repr=input_repr, embedding_size=embedding_size)

    # batch
    last_batch = filenames_num % batch_size
    batch_num = filenames_num // batch_size
    if last_batch > 0:
        batch_num += 1
    for batch_idx in range(batch_num):
        start_idx = batch_idx * batch_size
        end_idx = -1 if batch_idx == batch_num - 1 else (batch_idx + 1) * batch_size
        audio_list = []

        for filename in filenames[batch_idx * batch_size:(batch_idx + 1) * batch_size]:
            audio, sr = sf.read(audio_path + filename)
            audio_list.append(audio)

        tmp_emb_list, _ = openl3.get_audio_embedding(audio_list, sr, model=model)
        emb_list.extend(tmp_emb_list)
        print('Batch_idx:{}, start_idx:{}, end_idx:{}, emb_list size:{}'.format(batch_idx, start_idx, end_idx,
                                                                                len(emb_list)))

    ##Save the 2,000 filenames-embeddings into a numpy array
    dataset = tuple(zip(filenames, emb_list))
    np.save('ESC-50_openl3_' + content_type + '_' + input_repr + '_' + str(embedding_size) + '.npy', dataset)
    print('Features computed')


##Install OpenL3, an open-source Python library for computing deep audio and image embeddings. https://github.com/marl/openl3
##Once dowloaded ESC-50 dataset, update both directories and call the function
##I can also provide a link to the computed features if you email me at the address on the top.
if __name__ == '__main__':
    metadata = 'ESC-50-master/meta/esc50.csv'
    audio_path = 'ESC-50-master/audio/'
    compute_audio_embeddings(audio_path, metadata, batch_size=2000)
