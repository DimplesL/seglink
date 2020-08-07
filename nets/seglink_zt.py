import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.append('nets/')
import net_factory
import config
from tf_extended.seglink import generate_new_anchors, tf_get_all_seglink_gt
import pdb


class SegLinkNet(object):
    def __init__(self, inputs, weight_decay=None, data_format='NHWC',
                 weights_initializer=None, biases_initializer=None):
        self.inputs = inputs
        self.weight_decay = weight_decay
        self.feat_layers = config.feat_layers
        self.basenet_type = config.basenet_type
        self.data_format = data_format
        if weights_initializer is None:
            weights_initializer = tf.contrib.layers.xavier_initializer()
        if biases_initializer is None:
            biases_initializer = tf.zeros_initializer()
        self.weights_initializer = weights_initializer
        self.biases_initializer = biases_initializer
        self.is_training = config.is_training

        self._build_network()
        self.shapes = self.get_shapes()

    def get_shapes(self):
        shapes = {}

        for layer in self.end_points:
            shapes[layer] = tensor_shape(self.end_points[layer])[1:-1]
        return shapes

    def get_shape(self, name):
        return self.shapes[name]

    def _build_network(self):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            weights_initializer=self.weights_initializer,
                            biases_initializer=self.biases_initializer):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=self.data_format):
                with tf.variable_scope(self.basenet_type):
                    basenet, end_points = net_factory.get_basenet(self.basenet_type, self.inputs,
                                                                  is_training=self.is_training)

                with tf.variable_scope('extra_layers'):
                    self.net, self.end_points = self._add_extra_layers(basenet, end_points)

                with tf.variable_scope('seglink_layers'):
                    self._add_seglink_layers()

    def _add_extra_layers(self, inputs, end_points):
        # Additional SSD blocks.
        # conv6/7/8/9/10: 1x1 and 3x3 convolutions stride 2 (except lasts).
        # dilation is added
        net = slim.conv2d(inputs, 1024, [3, 3], rate=6, scope='fc6')
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
        end_points['conv5'] = net

        net = slim.conv2d(inputs, 256, [1, 1], scope='conv6_1')
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv6_2', padding='SAME')
        end_points['conv6'] = net

        net = slim.conv2d(net, 128, [1, 1], scope='conv7_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv7_2', padding='SAME')
        end_points['conv7'] = net

        net = slim.conv2d(net, 128, [1, 1], scope='conv8_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv8_2', padding='SAME')
        end_points['conv8'] = net

        net = slim.conv2d(net, 128, [1, 1], scope='conv9_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_2', padding='SAME')
        end_points['conv9'] = net

        if config.use_fpn:
            for level in range(8, 3, -1):  # build [P8~P4]
                end_points['conv%d' % level] = self.fusion_two_layer(C_i=end_points["conv%d" % level],
                                                                     P_j=end_points["conv%d" % (level + 1)],
                                                                     scope='Rebuild_conv%d' % level)

        if config.use_attention:
            # # 生成mask attention
            h, w = tf.shape(end_points['conv3'])[1], tf.shape(end_points['conv3'])[2]
            mask = self.fusion_three_layer(end_points['conv3'], end_points['conv4'], end_points['conv5'],
                                           scope='mask')
            # mask = F.relu(self.conv1_bn_mask(self.conv1_mask(mask)))
            mask = tf.nn.relu(slim.batch_norm(mask, is_training=config.is_training))
            # mask = self.deconv_mask(mask)
            upsample_mask = tf.image.resize_bilinear(mask, size=[h, w])
            mask = slim.conv2d(upsample_mask, num_outputs=512,
                               kernel_size=[3, 3], stride=1, rate=2)
            # self.conv2_mask
            mask = slim.conv2d(mask, num_outputs=2,
                               kernel_size=[3, 3], stride=1, rate=2)
            mask = tf.nn.relu(slim.batch_norm(mask, is_training=config.is_training))
            # self.conv3_mask
            mask = slim.conv2d(mask, num_outputs=2,
                               kernel_size=[3, 3], stride=1, rate=2)
            attention = tf.nn.softmax(mask)[:, :, :, 1:2]
            self.attention = attention
            end_points['attention'] = attention

            # 将attention 点乘到特征图
            for level in range(4, 7):
                attention = slim.max_pool2d(attention, kernel_size=2, stride=2, scope='attention%d' % level)
                broadcast = tf.broadcast_to(attention, tf.shape(end_points['conv%d' % level]))
                end_points['conv%d' % level] = end_points['conv%d' % level] * broadcast

        return net, end_points

    def fusion_three_layer(self, top, mid, bottom, scope):
        with tf.variable_scope(scope):
            h, w = tf.shape(mid)[1], tf.shape(mid)[2]
            C = 1024 if scope[-2] == '5' else 512
            top = slim.conv2d(top, C, [1, 1], scope='{}_conv_top'.format(scope))
            down_top = slim.max_pool2d(top, [2, 2], scope='{}_pool_top'.format(scope))

            upsample_bottom = tf.image.resize_bilinear(bottom, size=[h, w],
                                                       name='{}_up_sample'.format(scope))
            reduce_dim_c = slim.conv2d(upsample_bottom, num_outputs=C,
                                       kernel_size=[1, 1], stride=1,
                                       scope='{}_conv_bottom'.format(scope))
            add_f = 1.0 / 3 * down_top + 1.0 / 3 * mid + 1.0 / 3 * reduce_dim_c
            out = slim.conv2d(add_f, num_outputs=C, kernel_size=[3, 3], padding="SAME",
                              stride=1, scope="fuse_{}".format(scope))

        return out

    def fusion_two_layer(self, C_i, P_j, scope):
        with tf.variable_scope(scope):
            level_name = scope.split('_')[1]
            h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
            upsample_p = tf.image.resize_bilinear(P_j,
                                                  size=[h, w],
                                                  name='up_sample_' + level_name)
            reduce_dim_c = slim.conv2d(C_i,
                                       num_outputs=256,
                                       kernel_size=[1, 1], stride=1,
                                       scope='reduce_dim_' + level_name)

            add_f = 0.5 * upsample_p + 0.5 * reduce_dim_c
            out = slim.conv2d(add_f, num_outputs=256, kernel_size=[3, 3], padding="SAME",
                              stride=1, scope="fuse_{}".format(scope))

            return out

    def _build_seg_link_layer(self, layer_name):
        net = self.end_points[layer_name]
        batch_size, h, w = tensor_shape(net)[:-1]

        # if layer_name == 'conv4_3':
        if layer_name == config.feat_layers[0]:
            net = tf.nn.l2_normalize(net, -1) * 20

        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            # segment scores
            num_cls_pred = 2
            seg_scores = slim.conv2d(net, num_cls_pred, [3, 3], scope='seg_scores')
            # segment offsets
            num_offset_pred = 5
            seg_offsets = slim.conv2d(net, num_offset_pred, [3, 3], scope='seg_offsets')
            # within-layer link scores
            num_within_layer_link_scores_pred = 16
            within_layer_link_scores = slim.conv2d(net, num_within_layer_link_scores_pred, [3, 3],
                                                   scope='within_layer_link_scores')
            within_layer_link_scores = tf.reshape(within_layer_link_scores,
                                                  tensor_shape(within_layer_link_scores)[:-1] + [8, 2])
            # cross-layer link scores
            num_cross_layer_link_scores_pred = 8
            cross_layer_link_scores = None
            if layer_name != config.feat_layers[0]:
                cross_layer_link_scores = slim.conv2d(net, num_cross_layer_link_scores_pred, [3, 3],
                                                      scope='cross_layer_link_scores')
                cross_layer_link_scores = tf.reshape(cross_layer_link_scores,
                                                     tensor_shape(cross_layer_link_scores)[:-1] + [4, 2])

        return seg_scores, seg_offsets, within_layer_link_scores, cross_layer_link_scores

    def _add_seglink_layers(self):
        all_seg_scores = []
        all_seg_offsets = []
        all_within_layer_link_scores = []
        all_cross_layer_link_scores = []
        for layer_name in self.feat_layers:
            with tf.variable_scope(layer_name):
                seg_scores, seg_offsets, within_layer_link_scores, cross_layer_link_scores = self._build_seg_link_layer(
                    layer_name)
            all_seg_scores.append(seg_scores)
            all_seg_offsets.append(seg_offsets)
            all_within_layer_link_scores.append(within_layer_link_scores)
            all_cross_layer_link_scores.append(cross_layer_link_scores)
        self.seg_score_logits = reshape_and_concat(all_seg_scores)  # (batch_size, N, 2)
        self.seg_scores = slim.softmax(self.seg_score_logits)  # (batch_size, N, 2)
        self.seg_offsets = reshape_and_concat(all_seg_offsets)  # (batch_size, N, 5)
        self.cross_layer_link_scores = reshape_and_concat(all_cross_layer_link_scores)  # (batch_size, 8N, 2)
        self.within_layer_link_scores = reshape_and_concat(
            all_within_layer_link_scores)  # (batch_size, 4(N - N_conv4_3), 2)
        self.link_score_logits = tf.concat([self.within_layer_link_scores, self.cross_layer_link_scores], axis=1)
        self.link_scores = slim.softmax(self.link_score_logits)

    def refine_stage(self):
        new_anchors = tf.py_func(generate_new_anchors, [self.seg_offsets], tf.float32)
        config._generate_refine_anchors(new_anchors)

    def build_loss(self, seg_labels, seg_offsets, link_labels, attention_label, do_summary=True):
        batch_size = int(config.batch_size_per_gpu)

        # note that for label values in both seg_labels and link_labels:
        #    -1 stands for negative
        #     1 stands for positive
        #     0 stands for ignored
        def get_pos_and_neg_masks(labels):
            if config.train_with_ignored:
                pos_mask = labels >= 0
                neg_mask = tf.logical_not(pos_mask)
            else:
                pos_mask = tf.equal(labels, 1)
                neg_mask = tf.equal(labels, -1)

            return pos_mask, neg_mask

        def OHNM_single_image(scores, n_pos, neg_mask):
            """Online Hard Negative Mining.
                scores: the scores of being predicted as negative cls
                n_pos: the number of positive samples
                neg_mask: mask of negative samples
                Return:
                    the mask of selected negative samples.
                    if n_pos == 0, no negative samples will be selected.
            """

            def has_pos():
                n_neg = n_pos * config.max_neg_pos_ratio
                max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))
                n_neg = tf.minimum(n_neg, max_neg_entries)
                n_neg = tf.cast(n_neg, tf.int32)
                neg_conf = tf.boolean_mask(scores, neg_mask)
                vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
                threshold = vals[-1]  # a negtive value
                selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
                return tf.cast(selected_neg_mask, tf.float32)

            def no_pos():
                return tf.zeros_like(neg_mask, tf.float32)

            return tf.cond(n_pos > 0, has_pos, no_pos)

        def OHNM_batch(neg_conf, pos_mask, neg_mask):
            selected_neg_mask = []
            for image_idx in range(batch_size):
                image_neg_conf = neg_conf[image_idx, :]
                image_neg_mask = neg_mask[image_idx, :]
                image_pos_mask = pos_mask[image_idx, :]
                n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
                selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))

            selected_neg_mask = tf.stack(selected_neg_mask)
            selected_mask = tf.cast(pos_mask, tf.float32) + selected_neg_mask
            return selected_mask

        # OHNM on segments
        seg_neg_scores = self.seg_scores[:, :, 0]
        seg_pos_mask, seg_neg_mask = get_pos_and_neg_masks(seg_labels)
        seg_selected_mask = OHNM_batch(seg_neg_scores, seg_pos_mask, seg_neg_mask)
        n_seg_pos = tf.reduce_sum(tf.cast(seg_pos_mask, tf.float32))

        with tf.name_scope('seg_cls_loss'):
            def has_pos_OHEM():
                seg_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.seg_score_logits,
                    labels=tf.cast(seg_pos_mask, dtype=tf.int32))
                return tf.reduce_sum(seg_cls_loss * seg_selected_mask) / n_seg_pos

            def has_pos_FL(gamma=2, alpha=None):
                per_entry_cross_ent = (tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(seg_pos_mask, dtype=tf.int32), logits=self.seg_score_logits))
                prediction_probabilities = slim.softmax(self.seg_score_logits)[:, :, 1]
                labels = tf.cast(seg_pos_mask, tf.float32)
                p_t = ((labels * prediction_probabilities) +
                       ((1 - labels) * (1 - prediction_probabilities)))
                modulating_factor = tf.pow(1.0 - p_t, gamma)
                if alpha == None:
                    alpha_weight_factor = 1.0
                else:
                    alpha_weight_factor = (labels * alpha + (1 - labels) * (1 - alpha))

                focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                            per_entry_cross_ent * seg_selected_mask)

                return tf.reduce_sum(focal_cross_entropy_loss) / n_seg_pos

            def no_pos():
                return tf.constant(.0)

            if config.use_focal_loss:
                seg_cls_loss = tf.cond(n_seg_pos > 0, has_pos_FL, no_pos) * config.seg_cls_loss_weight
            else:
                seg_cls_loss = tf.cond(n_seg_pos > 0, has_pos_OHEM, no_pos) * config.seg_cls_loss_weight

            tf.add_to_collection(tf.GraphKeys.LOSSES, seg_cls_loss)

        def smooth_l1_loss(pred, target, weights):
            diff = pred - target
            abs_diff = tf.abs(diff)
            abs_diff_lt_1 = tf.less(abs_diff, 1)
            if len(target.shape) != len(weights.shape):
                loss = tf.reduce_sum(tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5), axis=2)
                return tf.reduce_sum(loss * tf.cast(weights, tf.float32))
            else:
                loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
                return tf.reduce_sum(loss * tf.cast(weights, tf.float32))

        with tf.name_scope('seg_loc_loss'):
            def has_pos():
                seg_loc_loss = smooth_l1_loss(self.seg_offsets, seg_offsets,
                                              seg_pos_mask) * config.seg_loc_loss_weight / n_seg_pos
                names = ['loc_cx_loss', 'loc_cy_loss', 'loc_w_loss', 'loc_h_loss', 'loc_theta_loss']
                sub_loc_losses = []
                from tensorflow.python.ops import control_flow_ops
                for idx, name in enumerate(names):
                    name_loss = smooth_l1_loss(self.seg_offsets[:, :, idx], seg_offsets[:, :, idx],
                                               seg_pos_mask) * config.seg_loc_loss_weight / n_seg_pos
                    name_loss = tf.identity(name_loss, name=name)
                    if do_summary:
                        tf.summary.scalar(name, name_loss)
                    sub_loc_losses.append(name_loss)
                seg_loc_loss = control_flow_ops.with_dependencies(sub_loc_losses, seg_loc_loss)
                return seg_loc_loss

            def no_pos():
                return tf.constant(.0)

            seg_loc_loss = tf.cond(n_seg_pos > 0, has_pos, no_pos)
            tf.add_to_collection(tf.GraphKeys.LOSSES, seg_loc_loss)

        link_neg_scores = self.link_scores[:, :, 0]
        link_pos_mask, link_neg_mask = get_pos_and_neg_masks(link_labels)
        link_selected_mask = OHNM_batch(link_neg_scores, link_pos_mask, link_neg_mask)
        n_link_pos = tf.reduce_sum(tf.cast(link_pos_mask, dtype=tf.float32))
        with tf.name_scope('link_cls_loss'):
            def has_pos_OHEM():
                link_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.link_score_logits,
                    labels=tf.cast(link_pos_mask, tf.int32))

                return tf.reduce_sum(link_cls_loss * link_selected_mask) / n_link_pos

            def has_pos_FL(gamma=2, alpha=0.75):
                per_entry_cross_ent = (tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(link_pos_mask, dtype=tf.int32), logits=self.link_score_logits))
                prediction_probabilities = slim.softmax(self.link_score_logits)[:, :, 1]

                labels = tf.cast(link_pos_mask, tf.float32)

                p_t = ((labels * prediction_probabilities) +
                       ((1 - labels) * (1 - prediction_probabilities)))
                modulating_factor = tf.pow(1.0 - p_t, gamma)

                alpha_weight_factor = (labels * alpha + (1 - labels) * (1 - alpha))

                focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)

                return tf.reduce_sum(focal_cross_entropy_loss) / n_link_pos

            def no_pos():
                return tf.constant(.0)

            # if config.use_focal_loss:
            #     link_cls_loss = tf.cond(n_link_pos > 0, has_pos_FL, no_pos) * config.link_cls_loss_weight
            # else:
            #     link_cls_loss = tf.cond(n_link_pos > 0, has_pos_OHEM, no_pos) * config.link_cls_loss_weight

            link_cls_loss = tf.cond(n_link_pos > 0, has_pos_OHEM, no_pos) * config.link_cls_loss_weight

            tf.add_to_collection(tf.GraphKeys.LOSSES, link_cls_loss)

        if config.use_attention:
            beta = 1 - tf.reduce_mean(attention_label)
            predicts = self.attention
            attention_loss = tf.reduce_mean(
                -1 * (beta * attention_label * tf.log(predicts + 1e-4) +
                      (1 - beta) * (1 - attention_label) * tf.log(1 - predicts + 1e-4))) * config.attention_loss_weight
            tf.add_to_collection(tf.GraphKeys.LOSSES, attention_loss)
        else:
            attention_loss = 0
        if do_summary:
            tf.summary.scalar('seg_cls_loss', seg_cls_loss)
            tf.summary.scalar('seg_loc_loss', seg_loc_loss)
            tf.summary.scalar('link_cls_loss', link_cls_loss)
            tf.summary.scalar('attention_loss', attention_loss)


def reshape_and_concat(tensors):
    def reshape(t):
        shape = tensor_shape(t)
        if len(shape) == 4:
            shape = (shape[0], -1, shape[-1])
            t = tf.reshape(t, shape)
        elif len(shape) == 5:
            shape = (shape[0], -1, shape[-2], shape[-1])
            t = tf.reshape(t, shape)
            t = tf.reshape(t, [shape[0], -1, shape[-1]])
        else:
            raise ValueError("invalid tensor shape: %s, shape = %s" % (t.name, shape))
        return t

    reshaped_tensors = [reshape(t) for t in tensors if t is not None]
    return tf.concat(reshaped_tensors, axis=1)


def tensor_shape(t):
    t.get_shape().assert_is_fully_defined()
    return t.get_shape().as_list()
