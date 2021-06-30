import nibabel as nib
import nilearn
from skimage import data, io, exposure, filters
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # fi = ants.image_read('./t2.nii.gz')
    # mi = ants.image_read('./spgr_unstrip.nii')
    # freg = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN')
    # fregimage = ants.apply_transforms(fixed=fi, moving=mi, transformlist=freg['fwdtransforms'])
    # fo = ants.resample_image_to_target(fregimage, mi, interp_type='nearestNeighbor')
    # ants.image_write(fo, 't2_resample.nii.gz')
    fix_path = 'spgr_unstrip.nii'
    move_path = 't2.nii.gz'
    save_path = 't2_reg_syn.nii.gz'
    exposure.equalize_hist()
    filters.roberts()
    # move_img = ants.resample_image(move_img, (245, 245, 160), True, 1)
    # ants.image_write(move_img, 't2_res.nii.gz')
    # outs = ants.registration(fix_img, move_img, type_of_transforme='SyN')
    # reg_img = outs['warpedmovout']
    # ants.image_write(reg_img, save_path)
