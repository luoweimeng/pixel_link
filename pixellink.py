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


class pixel_link_test(object):
    def __init__(self):
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


def load_pixel_link_model(checkpoint_dir, eval_image_height, eval_image_width,
                 pixel_conf_threshold, link_conf_threshold, updown_link_conf_threshold):

    global_step = slim.get_or_create_global_step()
    config_initialization(eval_image_height, eval_image_width,
                          pixel_conf_threshold, link_conf_threshold, updown_link_conf_threshold)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        # Construct the graph
        pl_net = pixel_link_test()

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(var_list=variables_to_restore)

        ckpt = checkpoint_dir
        print(ckpt)
        if ckpt:
            saver.restore(sess, ckpt)
            print(checkpoint_dir, "Restore Success!")

    return sess, pl_net


sess, pl_net = load_pixel_link_model("train/ic17_whole/model.ckpt-200000", 1280, 1280, 0.5, 0.7, 0.8)


class pixelLinkDetector(object):
    def __init__(self, image_path):
        self.image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # self.image_data, scale = resize_im(self.image_data, scale=768, max_scale=1280)


    def detect(self):
        """

        :return: list of bounding boxes 左上 右上 左下 右下
        """
        graph = tf.Graph()
        with graph.as_default():
            self.link_scores, self.pixel_scores, self.mask_vals = sess.run(
                [pl_net.net.link_pos_scores, pl_net.net.pixel_pos_scores, pl_net.masks],
                feed_dict={pl_net.image: self.image_data})


            def get_bboxes(mask):
                return pixel_link.mask_to_bboxes(mask, self.image_data.shape)


            image_idx = 0
            pixel_score = self.pixel_scores[image_idx, ...]
            mask = self.mask_vals[image_idx, ...]
            start_post_time = time.time()
            self.bboxes_det = get_bboxes(mask)

            bboxes_new_order = []
            for bbox in self.bboxes_det:
                bbox = bbox.reshape(4, 2)
                #find the index of upper-left (x+y is minimal)
                upper_left = np.argmin(map(lambda x: x[0] + x[1], bbox))
                upper_right = (upper_left + 1) % 4
                down_left = (upper_left + 3) % 4
                down_right = (upper_left + 2) % 4

                bboxes_new_order.append(bbox[[upper_left, upper_right, down_left, down_right]].reshape(-1))

            return bboxes_new_order


    def draw_bbox(self):
        def draw_bboxes(img, bboxes, color):
            for bbox in bboxes:
                points = np.reshape(bbox, [4, 2])

                cnts = util.img.points_to_contours(points)
                util.img.draw_contours(img, contours=cnts,
                                       idx=-1, color=color, border_width=2)

        draw_bboxes(self.image_data, self.bboxes_det, util.img.COLOR_RGB_RED)
        return self.image_data


    def draw_pixel_score(self):
        """

        :return: the map of pixel_score of original size: w * h
        """
        h, w, _ = self.image_data.shape

        # resize to original size
        def resize(img):
            return util.img.resize(img, size=(w, h),
                                   interpolation=cv2.INTER_NEAREST)
        pixel_score = resize(self.pixel_scores[0])
        return pixel_score


    def draw_link_score(self, i):
        """

        :param i: direction 0:左上 1:上 2:右上 3:左 4:右 5:左下 6:下 7:右下

        :return: the map of link_score at i th direction of original size: w * h
        """
        h, w, _ = self.image_data.shape

        # resize to original size
        def resize(img):
            return util.img.resize(img, size=(w, h),
                                   interpolation=cv2.INTER_NEAREST)

        return resize(self.link_scores[0, :, :, i])

    def draw_mask_vals(self):
        """

        :return: the map of mask values of original size: w * h
        """
        h, w, _ = self.image_data.shape

        # resize to original size
        def resize(img):
            return util.img.resize(img, size=(w, h),
                                   interpolation=cv2.INTER_NEAREST)

        return resize(self.mask_vals[0])


if __name__ == "__main__":
    pl = pixelLinkDetector("/Users/luoweimeng/Code/data/test/40.jpg")
    print pl.detect()

    # util.sit(pl.draw_bbox(), format='bgr', path='/Users/luoweimeng/Code/data/test_output/40.jpg')
