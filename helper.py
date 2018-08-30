import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def get_all_imgs(train_image_dir):
    img_path = os.path.join(train_image_dir,'images')
    images = glob(os.path.join(img_path,'*.*'))
#     masks = glob.glob(os.path.join(mask_path,'*.*'))
    return [image.split('/')[-1] for image in images]


def gen_batch_function(data_folder, image_shape=(256,256)):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    
    def make_image_gen(batch_size = 4):
        all_batches = get_all_imgs(data_folder)
#         print(all_batches)
        out_rgb = []
        out_mask = []
        img_path = os.path.join(data_folder,'images')
        mask_path = os.path.join(data_folder,'masks')
        ctr = 0
        while True and ctr <= len(all_batches):
            np.random.shuffle(all_batches)
            for c_img_id in all_batches:
                c_img = scipy.misc.imread(os.path.join(img_path,c_img_id))
                #c_img = cv2.cvtColor(c_img,cv2.COLOR_RGB2LUV)
                c_mask = cv2.imread(os.path.join(mask_path,c_img_id),cv2.IMREAD_GRAYSCALE)
                
                
                c_img = cv2.resize(c_img,image_shape,interpolation = cv2.INTER_AREA)
                c_mask = cv2.resize(c_mask,image_shape,interpolation = cv2.INTER_AREA)
                c_mask = np.reshape(c_mask,(c_mask.shape[0],c_mask.shape[1],-1))
                
                c_mask = c_mask > 0
                c_mask_final = np.concatenate((np.invert(c_mask),c_mask ), axis=2)
                out_rgb += [c_img]
                out_mask += [c_mask_final]
                if len(out_rgb)>=batch_size:
                    ctr+=batch_size
                    yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                    out_rgb, out_mask=[], []
    return make_image_gen


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'images', '*.jpg')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGB")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_smoke/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
