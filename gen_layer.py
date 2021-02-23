from glob import glob
import numpy as np
import tensorflow as tf
import json
import os

import makiflow
from makiflow.generators.segmentator.pathgenerator import DistributionBasedPathGen
from makiflow.generators.segmentator.map_methods import ComputePositivesPostMethod
from makiflow.generators.segmentator.map_methods import NormalizePostMethod, RGB2BGRPostMethod, SqueezeMaskPostMethod
from makiflow.generators.segmentator import (InputGenNumpyGetterLayer, data_reader_wrapper, data_resize_wrapper, 
                                             data_elastic_wrapper, AugmentationPostMethod, binary_masks_reader)


LABELSETID_IMAGEFILENAMES_JSON_PATH = 'generator_configs/labelsetid_imagefilenames.json'
LABELSETID_PROB_JSON_PATH = 'generator_configs/labelsetid_prob.json'


def create_generator(path_images):
    images = glob(os.path.join(path_images, '*'))
    assert len(images) != 0, f'Found no images at {path_images}'
    masks = map(lambda x: x.replace('images', 'masks'), images)
    masks = list(map(lambda x: x.replace('.bmp', ''), masks))
    image_mask_dict = dict(zip(images, masks))

    def read_json(path):
        with open(path, 'r') as f:
            return json.loads(f.read())

    groupid_image_dict = read_json(LABELSETID_IMAGEFILENAMES_JSON_PATH)
    groupid_prob_dict = read_json(LABELSETID_PROB_JSON_PATH)
    gen = DistributionBasedPathGen(
        image_mask_dict,
        groupid_image_dict,
        groupid_prob_dict
    ).next_element()
    gen = binary_masks_reader(gen, 8, (1024, 1024))
    gen = data_elastic_wrapper(
       gen, alpha=500, std=5, num_maps=20, noise_invert_scale=15, seed=None,
       img_inter='linear', mask_inter='nearest', border_mode='reflect_101',
       keep_old_data=False, prob=0.9
    )

    return gen


def get_gen_layer(data_gen_path, im_hw, batch_sz, prefetch_sz):
    path_images, path_masks = data_gen_path
    map_method = AugmentationPostMethod(
        use_rotation=True,
        angle_min=-60.0,
        angle_max=60.0,
        use_shift=False,
        dx_min=None,
        dx_max=None,
        dy_min=None,
        dy_max=None,
        use_zoom=True,
        zoom_min=0.8,
        zoom_max=1.2
    )
    # map_method = SqueezeMaskPostMethod()(map_method)
    # map_method = RGB2BGRPostMethod()(map_method)
    map_method = NormalizePostMethod(use_float64=True)(map_method)
    map_method = ComputePositivesPostMethod()(map_method)
    return InputGenNumpyGetterLayer(
        prefetch_size=prefetch_sz,
        batch_size=batch_sz,
        path_generator=create_generator(path_images),
        name='Input',
        map_operation=map_method,
        num_parallel_calls=5,
        mask_shape=(im_hw[0], im_hw[1], 8),
        image_shape=(im_hw[0], im_hw[1], 3)
    )


def main():
    makiflow.set_main_gpu(1)
    sess = tf.Session()

    import matplotlib.pyplot as plt

    # gen = create_generator(f'/raid/rustam/med_data/aspai_{1}/images', f'/raid/rustam/med_data/aspai_{1}/masks')
    # gen = Generator(f'/raid/rustam/med_data/aspai_{1}/images', f'/raid/rustam/med_data/aspai_{1}/masks')
    # gen = get_generator(f'/raid/rustam/med_data/aspai_{1}/images', f'/raid/rustam/med_data/aspai_{1}/masks')
    t = gen.get_data_tensor()

    gen.get_iterator()

    plt.figure(figsize=(10, 10))
    plt.imshow(sess.run(t)[0].astype(np.uint8))

    plt.figure(figsize=(10, 10))
    plt.imshow(sess.run(gen.get_iterator()['mask'])[0].astype(np.uint8))


