import os
import argparse
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from collections import OrderedDict
from skimage.transform import resize
from scipy.ndimage import map_coordinates
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.utilities.file_and_folder_operations import isfile, subfolders

def resample_data_or_seg(data, new_shape, is_seg=False, order=3, cval=0):
    if is_seg:
        return resize_segmentation(data, new_shape, order=order)
    else:
        return resize(data, new_shape, order=order, mode='edge', cval=cval, anti_aliasing=False)

def resample_patient(data, seg, original_spacing, target_spacing, order_data=3, order_seg=0):
    # Calculate new shape based on the new spacing
    new_shape = np.round(np.array(data.shape) * (original_spacing / target_spacing)).astype(int)
    data_resampled = resample_data_or_seg(data, new_shape, order=order_data)
    seg_resampled = resample_data_or_seg(seg, new_shape, is_seg=True, order=order_seg)
    return data_resampled, seg_resampled

def resample_tindex(fold, data_path):
    vol_path = os.path.join(data_path, fold, 'imaging.nii.gz')
    seg_path = os.path.join(data_path, fold, 'segmentation_sinus.nii.gz')
    new_vol_path = os.path.join(data_path, fold, 'imaging_r.nii.gz')
    new_seg_path = os.path.join(data_path, fold, 'segmentation_sinus_r.nii.gz')

    if not isfile(new_vol_path) or not isfile(new_seg_path):
        vol_img = nib.load(vol_path)
        seg_img = nib.load(seg_path)

        vol_data = vol_img.get_fdata()
        seg_data = seg_img.get_fdata()

        original_spacing = np.array([vol_img.header['pixdim'][1], vol_img.header['pixdim'][2], vol_img.header['pixdim'][3]])
        target_spacing = np.array([1.0, 1.0, 1.0])  # Example target spacing

        vol_resampled, seg_resampled = resample_patient(vol_data, seg_data, original_spacing, target_spacing)

        new_vol_img = nib.Nifti1Image(vol_resampled, vol_img.affine)
        nib.save(new_vol_img, new_vol_path)

        new_seg_img = nib.Nifti1Image(seg_resampled, seg_img.affine)
        nib.save(new_seg_img, new_seg_path)

        print('Save complete:', new_vol_path)
        print('Save complete:', new_seg_path)
    else:
        print('Files already exist:', new_vol_path, new_seg_path)

def main(args):
    data_path = args.data_path
    folders = subfolders(data_path, join=False)
    with Pool(args.num_processes) as pool:
        pool.starmap(resample_tindex, [(fold, data_path) for fold in folders])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resampling.")
    parser.add_argument('--data_path', type=str, required=True, help='Base path for the data containing imaging files')
    parser.add_argument('--num_processes', type=int, default=3, help='Number of parallel processes to use for processing')
    args = parser.parse_args()
    main(args)