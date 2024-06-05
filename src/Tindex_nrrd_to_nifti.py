#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###############################
##################### nnunet/Tindex_nrrd_to_nifti.py
###############################
#import nibabel as nib
import SimpleITK as sitk
import nrrd
import numpy as np
def nrrd_to_nii(ori_path, seg_path, ori_save_path, seg_save_path):
    img_itk = sitk.ReadImage(ori_path)
    img_npy = sitk.GetArrayFromImage(img_itk)
    spacing = img_itk.GetSpacing()
    origin = img_itk.GetOrigin()
    direction = img_itk.GetDirection()

    seg = nrrd.read(seg_path)
    seg_data = seg[0]
    seg_header = seg[1]

    if len(seg_data.shape)==4:
        raw = np.zeros(seg_data[0].shape, dtype='uint8')

        # kidney:1, mass:2, cyst:3, sinus:4, background:0
        for i in range(np.shape(seg_data)[0]):
            head = 'Segment'+str(i)+'_Name'
            if 'kidney' in seg_header[head]:
                raw[np.where(seg_data[i] == 1)] = 1
            elif 'mass' in seg_header[head]:
                raw[np.where(seg_data[i] == 1)] = 2
            elif 'cyst' in seg_header[head]:
                raw[np.where(seg_data[i] == 1)] = 3

        raw = np.moveaxis(raw, [0,1,2], [2,1,0])

        head_list = seg_header['Segmentation_ReferenceImageExtentOffset'].split(' ')
        head_list = list(map(int, head_list))
        head_list = head_list[::-1]

        seg_raw = np.zeros(img_npy.shape, dtype='uint8')
        seg_raw[
                head_list[0]:head_list[0]+raw.shape[0],
                head_list[1]:head_list[1]+raw.shape[1],
                head_list[2]:head_list[2]+raw.shape[2]] = raw


    elif len(seg_data.shape)==3:
        raw = np.zeros(seg_data.shape, dtype='uint8')
        # kidney:1, mass:2, cyst:3, sinus:4, background:0
        for i in range(100):
            try:
                head = 'Segment'+str(i)+'_Name'
                Value = 'Segment'+str(i)+'_LabelValue'

                name_li=['kidney', 'Rk', 'Lk']
                if seg_header[head] in name_li:
                    raw[np.where(seg_data == int(seg_header[Value]))] = 1

                name_li=['mass', 'Rsrm', 'Lsrm']
                if seg_header[head] in name_li:
                    raw[np.where(seg_data == int(seg_header[Value]))] = 2

                name_li=['cyst', 'Rcyst', 'Lcyst']
                if seg_header[head] in name_li:
                    raw[np.where(seg_data == int(seg_header[Value]))] = 3

            except:
                pass
        raw = np.moveaxis(raw, [0,1,2], [2,1,0])
        seg_raw = raw.copy()


    seg_itk_new = sitk.GetImageFromArray(seg_raw)
    seg_itk_new.SetSpacing(spacing)
    seg_itk_new.SetOrigin(origin)
    seg_itk_new.SetDirection(direction)

    # original, seg nifti file save
    sitk.WriteImage(img_itk, ori_save_path)
    sitk.WriteImage(seg_itk_new, seg_save_path)
    
