import numpy as np

from umcglib.augment import augment, random_crop
from umcglib.images import center_crop_n
from umcglib.utils import apply_parallel, tsprint


def datagenerator(
    input_images: np.ndarray,
    output_images: np.ndarray,
    batch_size=5,
    shuffle=False,
    crop_shape=None,
    crop_method=None,
    normalization=None,
    augmentation=True,
    rotation_freq=0.2,
    tilt_freq=0.1,
    noise_freq=0.3,
    noise_mult=1e-3,
    mirror_freq=0.5,
):
    """
    Returns a (training) generator for use with model.fit().

    Parameters:
    input_modalities: List of modalty names to include.
    output_modalities: Names of the target modalities.
    batch_size: Number of images per batch (default: all).
    indexes: Only use the specified image indexes.
    shuffle: Shuffle the lists of indexes once at the beginning.
    augmentation: Apply augmentation or not (bool).
    """

    num_rows = len(input_images)
    indexes = list(range(num_rows))

    if batch_size == None:
        batch_size = len(indexes)

    idx = 0

    # Prepare empty batch placeholder with named inputs and outputs
    input_batch = np.zeros((batch_size,) + crop_shape + (3,))
    output_batch = np.zeros((batch_size,) + crop_shape + (1,))

    # Loop infinitely to keep generating batches
    while True:
        # Prepare each observation in a batch
        for batch_idx in range(batch_size):
            # Shuffle the order of images if all indexes have been seen
            if idx == 0 and shuffle:
                np.random.shuffle(indexes)

            current_index = indexes[idx]

            img_crop = input_images[current_index]
            seg_crop = output_images[current_index]

            if crop_method == "random":
                # Crop img and seg simultaneously so we get the same crop
                img_crop, seg_crop = random_crop(
                    img=img_crop, label=seg_crop, shape=crop_shape
                )

            if crop_method == "center":
                img_crop = center_crop_n(img_crop, crop_shape)
                seg_crop = center_crop_n(seg_crop, crop_shape)

            if augmentation:
                img_crop, seg_crop = augment(
                    img_crop,
                    seg_crop,
                    noise_chance=noise_freq,
                    noise_mult_max=noise_mult,
                    rotate_chance=rotation_freq,
                    tilt_chance=tilt_freq,
                    mirror_chance=mirror_freq,
                )

            input_batch[batch_idx] = img_crop
            output_batch[batch_idx] = seg_crop

            # Increase the current index and modulo by the number of rows
            #  so that we stay within bounds
            idx = (idx + 1) % len(indexes)

        yield input_batch, output_batch


def crop_and_augment(
    x,
    crop_shape=None,
    crop_method=None,
    augmentation=True,
    rotation_freq=0.2,
    tilt_freq=0.0,
    noise_freq=0.3,
    noise_mult=5e-2,
    flip_freq=0.5,
):

    if len(x) == 4:
        img, clinvars, interval, target = x
        clinvars_copy = np.copy(clinvars)
    else:
        img, interval, target = x
    img_crop = np.copy(img)
    interval_copy = np.copy(interval)
    
    #print("IC:", interval_copy, "CC", clinvars_copy)

    if crop_method == "random":
        # Crop img and seg simultaneously so we get the same crop
        img_crop, seg_crop = random_crop(img=img, label=seg, shape=crop_shape)

    if crop_method == "center":
        img_crop = center_crop_n(img, crop_shape)
        seg_crop = center_crop_n(seg, crop_shape)

    if augmentation:
        img_crop, seg_crop = augment(
            img_crop,
            img_crop[..., :1],
            noise_chance=noise_freq,
            noise_mult_max=noise_mult,
            rotate_chance=rotation_freq,
            tilt_chance=tilt_freq,
            mirror_chance=flip_freq,
        )
        
        #clinvars_copy += np.random.normal(0, 0.25, size=clinvars_copy.shape)
        
        #if target > 0.5:
        #    interval_copy = interval_copy + np.random.uniform(0, 2) # Between now and 2 years after
        #else:
        #    interval_copy = interval_copy - np.random.uniform(0, interval_copy) # Between scan and follow up
    if len(x) == 4:
        return img_crop, clinvars, interval, target
    else:
        return img_crop, interval, target

def balanced_datagenerator(
    input_images: np.ndarray,
    targets: np.ndarray,
    clinvars: np.ndarray = None,
    time_intervals: np.ndarray = None,
    batch_size=5,
    shuffle=False,
    crop_shape=None,
    **prep_args
):
    """
    Returns a (training) generator for use with model.fit().

    Parameters:
    input_modalities: List of modalty names to include.
    output_modalities: Names of the target modalities.
    batch_size: Number of images per batch (default: all).
    indexes: Only use the specified image indexes.
    shuffle: Shuffle the lists of indexes once at the beginning.
    augmentation: Apply augmentation or not (bool).
    """

    num_rows, *input_shape, num_channels = input_images.shape
    num_params = 0
    if clinvars is not None:
        use_clinvars = True
        num_params = clinvars.shape[1]
    else: 
        use_clinvars = False
        clinvars = np.zeros((num_rows, num_params))

    # Get indexes of positive samples
    tsprint("Getting positive indexes..")
    print(np.nonzero(targets))
    pos_idxs = list(np.nonzero(targets)[0])
    tsprint("Getting negative indexes..")
    neg_idxs = [i for i in range(num_rows) if i not in pos_idxs]

    num_pos = len(pos_idxs)
    num_neg = len(neg_idxs)

    if batch_size == None:
        raise ValueError("Batch size cannot be None in balanced generator.")

    # Keep track separately of the current indexes in the positive and negative
    # observations
    pos_idx = 0
    neg_idx = 0

    # Loop infinitely to keep generating batches
    while True:
        # Prepare each observation in a batch
        batch_raw = []
        for batch_idx in range(batch_size):
            #tsprint("BIDX:", batch_idx)
            # Shuffle the order of images if all indexes have been seen
            if pos_idx == 0 and shuffle:
                np.random.shuffle(pos_idxs)
            if neg_idx == 0 and shuffle:
                np.random.shuffle(neg_idxs)

            # Randomly choose if we add a positive or negative example
            sample_class = np.random.choice(["pos", "neg"])
            #tsprint("sample_class:", sample_class)

            if sample_class == "pos":
                sample_idx = pos_idxs[pos_idx]
                if use_clinvars:
                    batch_raw.append((input_images[sample_idx], clinvars[sample_idx], time_intervals[sample_idx], targets[sample_idx]))
                else:
                    batch_raw.append((input_images[sample_idx], time_intervals[sample_idx], targets[sample_idx]))
                pos_idx = (pos_idx + 1) % num_pos

            if sample_class == "neg":
                sample_idx = neg_idxs[neg_idx]
                if use_clinvars:
                    batch_raw.append((input_images[sample_idx], clinvars[sample_idx], time_intervals[sample_idx], targets[sample_idx]))
                else:
                    batch_raw.append((input_images[sample_idx], time_intervals[sample_idx], targets[sample_idx]))
                neg_idx = (neg_idx + 1) % num_neg

        # TODO: Augment clinical parameters?
        batch_prepped = apply_parallel(
            batch_raw, crop_and_augment, crop_shape=crop_shape, **prep_args
        )
        # tsprint("Done prepping batch")
        input_batch = np.asarray([b[0] for b in batch_prepped], dtype=np.float32)
        if use_clinvars:
            clinvar_batch = np.asarray([b[1] for b in batch_prepped], dtype=np.float32)
            intveral_batch = np.asarray([b[2] for b in batch_prepped], dtype=np.float32)
            output_batch = np.asarray([b[3] for b in batch_prepped], dtype=np.float32)
            yield (input_batch, clinvar_batch, intveral_batch), output_batch
        else:
            intveral_batch = np.asarray([b[1] for b in batch_prepped], dtype=np.float32)
            output_batch = np.asarray([b[2] for b in batch_prepped], dtype=np.float32)
            yield (input_batch, intveral_batch), output_batch
