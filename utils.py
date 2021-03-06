import os 
import glob
import numpy as np


def data_decompose(fpath, save_dir='./dataset_sample'):
    """[summary]

    Args:
        fpath ([type]): [description]
    """
    os.makedirs(save_dir,exist_ok=True)


    data = np.load(fpath)
    save_base_name = fpath.split('/')[-1].split('_')[0] # like 1-dust, 10-smoke

    for i in range(len(data)):
        save_sample = data[i]
        save_name = f'{save_base_name}_{i:05}.npy'
        np.save(os.path.join(save_dir, save_name), save_sample)

    #return data


if __name__ == '__main__':
    #fpath = './dataset/1-dust_labeled_spaces_img.npy'
    root_path = '/var/datasets/ParticleDatasetFSR/ParticleDatasetFSR/'
    save_dir = '/var/datasets/ParticleDatasetFSR/PDFSR_sample/'
    fpath_list = glob.glob(root_path+'*img.npy')

    for fpath in fpath_list:
        fpath= fpath.replace('\\', '/')
    
        data_decompose(fpath, save_dir) 
    print('done')