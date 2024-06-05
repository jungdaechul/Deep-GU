import os
import shutil
import argparse
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import binary_erosion
from skimage.measure import label
from skimage.morphology import convex_hull_image
from batchgenerators.utilities.file_and_folder_operations import isfile, subfolders

def get_largest_connected_component(segmentation):
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

def refine_kidney_segmentation(seg1, seg2, seg_ori_data):
    seg_dataset = []
    # seg1, seg2 : left, right kidney
    for seg_kid in [seg1, seg2]:
        x, y, z = np.where(seg_kid == 1)
        x_min = max(min(x) - 5, 0)
        y_min = max(min(y) - 5, 0)
        z_min = max(min(z) - 5, 0)
        x_max = min(max(x) + 5, seg_kid.shape[0])
        y_max = min(max(y) + 5, seg_kid.shape[1])
        z_max = min(max(z) + 5, seg_kid.shape[2])
        
        # crop kidney
        seg_kid_crop = seg_kid[x_min:x_max, y_min:y_max, z_min:z_max]
        crop_result = convex_hull_image(seg_kid_crop.astype('uint8'))
        
        result = np.zeros(seg_kid.shape)
        result[x_min:x_max, y_min:y_max, z_min:z_max] = crop_result
        
        seg_result = binary_erosion(result, iterations=3)
        seg_result *= np.isin(seg_ori_data, [0, 4])
        seg_result = get_largest_connected_component(seg_result)
        seg_dataset.append(seg_result)
    
    seg_new_data = np.zeros(seg_kid.shape)
    seg_new_data[seg_dataset[0] == 1] = 1
    seg_new_data[seg_dataset[1] == 1] = 1
    
    return seg_new_data

# kidney:1, tumor:2, cyst:3, sinus:4
def process_segmentation(seg_path, save_path):
    try:
        if not isfile(save_path):
            seg = nib.load(seg_path)
            seg_data = seg.get_fdata()
            seg_ori_data = seg_data.copy()
            seg_data[seg_data != 1] = 0

            labels = label(seg_data)
            region_list = [len(np.where(labels == b)[0]) for b in range(1, len(np.unique(labels)))]
            region_list_sort = np.argsort(region_list)

            seg1 = labels == region_list_sort[-1] + 1
            seg2 = labels == region_list_sort[-2] + 1

            result = refine_kidney_segmentation(seg1, seg2, seg_ori_data)
            seg_ori_data[result == 1] = 4
            seg_ori_data[seg_ori_data == 1] = 0
            seg_ori_data[seg1 == 1] = 1
            seg_ori_data[seg2 == 1] = 1

            vol = nib.Nifti1Image(seg_ori_data, seg.affine)
            nib.save(vol, save_path)
        print(f"Processed and saved: {save_path}")
    except Exception as e:
        print(f"Failed to process {seg_path}: {e}")

def main(args):
    data_path = args.data_path
    folders = subfolders(data_path, join=False)
    seg_path = [os.path.join(data_path, fold, 'segmentation.nii.gz') for fold in folders]
    save_path = [os.path.join(data_path, fold, 'segmentation_sinus.nii.gz') for fold in folders]

    with Pool(args.num_processes) as p:
        p.starmap_async(process_segmentation, zip(seg_path, save_path)).get()
        p.close()
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinus extraction.")
    parser.add_argument('--data_path', type=str, required=True, help='Base path for the data')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    args = parser.parse_args()
    main(args)