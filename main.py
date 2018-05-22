import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import argparse

FREEZE_GRAPH = False
KEEP_PROB = 0.6
LEARNING_RATE = 4e-5
EPOCHS = 20
BATCH_SIZE = 8
BETA = 2.5e-2


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    vgg_input_tensor_name = 'image_input:0'
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)

    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer3_out_tensor = graph.get_tensor_by_name(
        vgg_layer3_out_tensor_name)

    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer4_out_tensor = graph.get_tensor_by_name(
        vgg_layer4_out_tensor_name)

    vgg_layer7_out_tensor_name = 'layer7_out:0'
    vgg_layer7_out_tensor = graph.get_tensor_by_name(
        vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    if FREEZE_GRAPH:
        vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
        vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
        vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)

    layer7_out_1x1 = tf.layers.conv2d(
        vgg_layer7_out, num_classes, (1, 1), padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-2),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='my_layer7_out_1x1')

    layer4_input1 = tf.layers.conv2d_transpose(
        layer7_out_1x1, num_classes, (4, 4), (2, 2), padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-2),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='my_layer4_input1')
    layer4_input2 = tf.layers.conv2d(
        vgg_layer4_out, num_classes, (1, 1), padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-2),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='my_layer4_input2')

    layer4_output = tf.add(layer4_input1, layer4_input2,
                           name='my_layer4_output')

    layer3_input1 = tf.layers.conv2d_transpose(
        layer4_output, num_classes, (4, 4), (2, 2), padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-2),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='my_layer3_input1')
    layer3_input2 = tf.layers.conv2d(
        vgg_layer3_out, num_classes, (1, 1), padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-2),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='my_layer3_input2')

    layer3_output = tf.add(layer3_input1, layer3_input2,
                           name='my_layer3_output')

    nn_final_layer = tf.layers.conv2d_transpose(
        layer3_output, num_classes, (16, 16), (8, 8), padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=1e-2),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
        name='my_nn_final_layer')

    return nn_final_layer


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes),
                        name='my_logits_reshape')
    correct_label = tf.reshape(
        correct_label, (-1, num_classes), name='my_correct_labels_reshape')

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=correct_label, name='my_cross_entropy')
    mean_cross_entropy = tf.reduce_mean(
        cross_entropy, name='my_mean_cross_entropy')

    if FREEZE_GRAPH:
        trainables = []
        for variable in tf.trainable_variables():
            if 'my_' in variable.name or 'beta' in variable.name:
                trainables.append(variable)
                
        regularizer = tf.add_n([tf.nn.l2_loss(v) for v in trainables]) * BETA

        loss = tf.reduce_mean(mean_cross_entropy +
                              regularizer, name='my_final_loss')
        
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='my_optmizer')
        train_op = opt.minimize(loss, var_list=trainables, name="training_operation")
    else:
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='my_optmizer')
        train_op = opt.minimize(mean_cross_entropy, name="training_operation")
    return logits, train_op, mean_cross_entropy


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    loss_arr = []
    save_path = None
    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch+1))
        for X_batch, y_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
                input_image: X_batch,
                correct_label: y_batch,
                # can be between 0 and 1 during training
                keep_prob: KEEP_PROB,
                learning_rate: LEARNING_RATE
            })
        loss_arr.append(loss)
        print('Loss: {:.3f}'.format(loss))

    return save_path


tests.test_train_nn(train_nn)


def run():
    global FREEZE_GRAPH
    global KEEP_PROB
    global LEARNING_RATE
    global EPOCHS
    global BATCH_SIZE
    global BETA

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        nargs='?',
        default=EPOCHS,
        help='Number of epochs.'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        nargs='?',
        default=LEARNING_RATE,
        help='Learning rate'
    )

    parser.add_argument(
        '-kp',
        '--keep_probability',
        type=float,
        nargs='?',
        default=KEEP_PROB,
        help='Keep probability for dropout'
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        nargs='?',
        default=BATCH_SIZE,
        help='Batch size.'
    )
    parser.add_argument(
        '-beta',
        '--beta',
        type=float,
        nargs='?',
        default=BETA,
        help='Beta value of loss regularizer.'
    )

    args = parser.parse_args()
    print('\nArguments passed: ', args)

    FREEZE_GRAPH = True
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    KEEP_PROB = args.keep_probability
    BATCH_SIZE = args.batch_size
    BETA = args.beta

    print('\nTraining with epochs:', EPOCHS, 'learning rate:',
          LEARNING_RATE, 'keep_prob:', KEEP_PROB, 'batch_size:', BATCH_SIZE)

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = EPOCHS
        batch_size = BATCH_SIZE

        label = tf.placeholder(
            tf.int32, [None, None, None, num_classes], name='my_label')
        learning_rate = tf.placeholder(tf.float32, name='my_learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(
            sess, vgg_path)

        nn_last_layer = layers(
            vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(
            nn_last_layer, label, learning_rate, num_classes)

        if FREEZE_GRAPH:
            my_variable_initializers = [
                var.initializer for var in tf.global_variables() if 'my_' in var.name or
                'beta' in var.name
            ]
            sess.run(my_variable_initializers)
        else:
            sess.run(tf.global_variables_initializer())

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_image, label, keep_prob, learning_rate)

        helper.save_inference_samples(
            runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    tf.reset_default_graph()
    run()
