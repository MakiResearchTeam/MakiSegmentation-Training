from sklearn.utils import shuffle
from glob import glob
import os
import numpy as np
import makiflow
from makiflow.generators.pipeline.gen_base import PathGenerator
from makiflow.generators.segmentator.map_methods import SegmentIterator
from makiflow.generators.segmentator.map_methods import LoadDataMethod, ComputePositivesPostMethod, ResizePostMethod
from makiflow.generators.segmentator.map_methods import NormalizePostMethod, RGB2BGRPostMethod, SqueezeMaskPostMethod
from makiflow.generators.segmentator.gen_layers import InputGenLayer
import tensorflow as tf

from makiflow.generators.segmentator import (InputGenNumpyGetterLayer, data_reader_wrapper, data_resize_wrapper, 
                                             data_elastic_wrapper, AugmentationPostMethod)


class Generator(PathGenerator):
    def __init__(self, path_images, path_masks):
        self.images = glob(os.path.join(path_images, '*.bmp'))
        self.masks = glob(os.path.join(path_masks, '*.bmp'))
        for _ in range(50):
            self.images, self.masks = shuffle(self.images, self.masks)

    def next_element(self):
        index = 0
        while True:
            index = np.random.randint(low=0, high=len(self.images))
            
            el = {
                SegmentIterator.IMAGE: self.images[index],
                SegmentIterator.MASK: self.masks[index]
            }
            
            yield el


def create_generator(path_images, path_masks):
    gen = Generator(path_images, path_masks).next_element()
    gen = data_reader_wrapper(gen, use_bgr2rgb=False)
    gen = data_resize_wrapper(gen, resize_to=(1024,1024))
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
        use_zoom=False,
        zoom_min=0.9,
        zoom_max=1.1
    )
    map_method = SqueezeMaskPostMethod()(map_method)
    #map_method = RGB2BGRPostMethod()(map_method)
    map_method = NormalizePostMethod(use_float64=True)(map_method)
    map_method = ComputePositivesPostMethod()(map_method)
    return InputGenNumpyGetterLayer(
        prefetch_size=prefetch_sz,
        batch_size=batch_sz,
        path_generator=create_generator(path_images, path_masks),
        name='Input',
        map_operation=map_method,
        num_parallel_calls=5,
        mask_shape=im_hw,
        image_shape=(im_hw[0], im_hw[1], 3)
    )


def main():
    makiflow.set_main_gpu(1)
    sess = tf.Session()
    
    import matplotlib.pyplot as plt

    #gen = create_generator(f'/raid/rustam/med_data/aspai_{1}/images', f'/raid/rustam/med_data/aspai_{1}/masks')
    #gen = Generator(f'/raid/rustam/med_data/aspai_{1}/images', f'/raid/rustam/med_data/aspai_{1}/masks')
    gen = get_generator(f'/raid/rustam/med_data/aspai_{1}/images', f'/raid/rustam/med_data/aspai_{1}/masks')
    t = gen.get_data_tensor()

    gen.get_iterator()


    plt.figure(figsize=(10, 10))
    plt.imshow(sess.run(t)[0].astype(np.uint8))


    plt.figure(figsize=(10, 10))
    plt.imshow(sess.run(gen.get_iterator()['mask'])[0].astype(np.uint8))


