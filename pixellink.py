# -*- coding: utf-8 -*-
#!/usr/bin/env python
#title           :
#description     : Define the class of pixelLinkDetector
#author          :luoweimeng
#date            :2018/6/22
#version         :
#usage           :
#notes           :
#python_version  :2.7 
#==============================================================================

import tensorflow as tf
import numpy as np
from preprocessing import ssd_vgg_preprocessing
import cv2
import pixel_link
from nets import pixel_link_symbol
import time
import config
import util

slim = tf.contrib.slim


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f


def config_initialization(eval_image_height, eval_image_width,
                          pixel_conf_threshold, link_conf_threshold, updown_link_conf_threshold):
    # image shape and feature layers shape inference

    # print(FLAGS.updown_link_conf_threshold)
    image_shape = (eval_image_height, eval_image_width)
    print(image_shape)

    # if not FLAGS.dataset_dir:
        # raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)

    config.init_config(image_shape,
                       batch_size=1,
                       pixel_conf_threshold=pixel_conf_threshold,
                       link_conf_threshold=link_conf_threshold,
                       updown_link_conf_threshold=updown_link_conf_threshold,
                       num_gpus=1,
                       )

class pixelLinkDetector(object):
    def __init__(self, checkpoint_dir, eval_image_height, eval_image_width,
                 pixel_conf_threshold, link_conf_threshold, updown_link_conf_threshold):

        config_initialization(eval_image_height, eval_image_width,
                              pixel_conf_threshold, link_conf_threshold, updown_link_conf_threshold)
        global_step = slim.get_or_create_global_step()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            # Construct the graph
            with tf.name_scope('evaluation_%dx%d' % (eval_image_height, eval_image_width)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    self.image = tf.placeholder(dtype=tf.int32, shape=[None, None, 3])
                    image_shape = tf.placeholder(dtype=tf.int32, shape=[3, ])
                    processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(self.image, None, None, None, None,
                                                                                         out_shape=(768, 768),
                                                                                         data_format='NHWC',
                                                                                         is_training=False)
                    b_image = tf.expand_dims(processed_image, axis=0)

                    # build model and loss
                    self.net = pixel_link_symbol.PixelLinkNet(b_image, is_training=False)
                    self.masks = pixel_link.tf_decode_score_map_to_mask_in_batch(
                        self.net.pixel_pos_scores, self.net.link_pos_scores)

            variables_to_restore = slim.get_variables_to_restore()
            self.saver = tf.train.Saver(var_list=variables_to_restore)

            ckpt = checkpoint_dir
            print(ckpt)
            if ckpt:
                self.saver.restore(self.sess, ckpt)
                print(checkpoint_dir, "Restore Success!")

    def detect(self, image_path):
        with self.graph.as_default():
            self.image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)

            self.image_data, scale = resize_im(self.image_data, scale=768, max_scale=1280)

            link_scores, pixel_scores, mask_vals = self.sess.run(
                [self.net.link_pos_scores, self.net.pixel_pos_scores, self.masks],
                feed_dict={self.image: self.image_data})

            h, w, _ = self.image_data.shape


            def get_bboxes(mask):
                return pixel_link.mask_to_bboxes(mask, self.image_data.shape)


            image_idx = 0
            pixel_score = pixel_scores[image_idx, ...]
            mask = mask_vals[image_idx, ...]
            start_post_time = time.time()
            self.bboxes_det = get_bboxes(mask)

            # mask = resize(mask)
            # pixel_score = resize(pixel_score)

            return self.bboxes_det

    def draw_bbox(self):
        def draw_bboxes(img, bboxes, color):
            for bbox in bboxes:
                points = np.reshape(bbox, [4, 2])

                cnts = util.img.points_to_contours(points)
                util.img.draw_contours(img, contours=cnts,
                                       idx=-1, color=color, border_width=4)

        draw_bboxes(self.image_data, self.bboxes_det, util.img.COLOR_RGB_RED)
        return self.image_data


if __name__ == "__main__":
    pl = pixelLinkDetector("train/ic17_whole/model.ckpt-200000", 768, 768, 0.5, 0.7, 0.8)
    print pl.detect("/Users/luoweimeng/Code/data/test/36.jpg")

    util.sit(pl.draw_bbox(), format='bgr', path='/Users/luoweimeng/Code/data/test_output/36.jpg')