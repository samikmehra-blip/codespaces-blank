from astropy.io import fits
import numpy as np

def process_spherex_pair(before_fits_path, after_fits_path):
    with fits.open(before_fits_path) as hdul_b, fits.open(after_fits_path) as hdul_a:
        img_before = hdul_b[1].data.astype(np.float32)
        img_after = hdul_a[1].data.astype(np.float32)
        
    img_before = np.nan_to_num(img_before)
    img_after = np.nan_to_num(img_after)
    
    img_diff = img_after - img_before
    tensor_3ch = np.stack([img_before, img_after, img_diff], axis=0)
    return tensor_3ch

def build_dataset_arrays():
    cutouts_list = []
    labels_list = []

    sample_pairs = [
        ('cutout_before.fits', 'cutout_after.fits', 1),
    ]

    for before_path, after_path, label in sample_pairs:
        tensor_3ch = process_spherex_pair(before_path, after_path)
        cutouts_list.append(tensor_3ch)
        labels_list.append([label])

    X_data = np.array(cutouts_list, dtype=np.float32)
    y_data = np.array(labels_list, dtype=np.float32)

    np.save('spherex_train_cutouts.npy', X_data)
    np.save('spherex_train_labels.npy', y_data)
    print("Generated .npy training arrays")

if __name__ == "__main__":
    build_dataset_arrays()