from dependencies.facenet.src import facenet
from dependencies.facenet.src.models import inception_resnet_v1 as tf_mdl
import tensorflow as tf
import torch
import json
import os

from models.inception_resnet_v1 import InceptionResNetV1


def import_tf_params(tf_mdl_dir, sess):

    print('\nLoading tensorflow model\n')
    facenet.load_model(tf_mdl_dir)

    print('\nGetting model weights\n')
    tf_layers = tf.trainable_variables()
    tf_params = sess.run(tf_layers)

    tf_shapes = [p.shape for p in tf_params]
    tf_layers = [l.name for l in tf_layers]

    with open(os.path.join(tf_mdl_dir, 'layer_description.json'), 'w') as f:
        json.dump({l: s for l, s in zip(tf_layers, tf_shapes)}, f)

    return tf_layers, tf_params, tf_shapes


def get_layer_indices(layer_lookup, tf_layers):
    layer_inds = {}
    for name, value in layer_lookup.items():
        layer_inds[name] = value + [[i for i, n in enumerate(tf_layers) if value[0] in n]]
    return layer_inds


def load_tf_batchNorm(weights, layer):
    layer.bias.data = torch.tensor(weights[0]).view(layer.bias.data.shape)
    layer.weight.data = torch.ones_like(layer.weight.data)
    layer.running_mean = torch.tensor(weights[1]).view(layer.running_mean.shape)
    layer.running_var = torch.tensor(weights[2]).view(layer.running_var.shape)


def load_tf_conv2d(weights, layer):
    if isinstance(weights, list):
        if len(weights) == 2:
            layer.bias.data = (
                torch.tensor(weights[1])
                    .view(layer.bias.data.shape)
            )
        weights = weights[0]
    layer.weight.data = (
        torch.tensor(weights)
            .permute(3, 2, 0, 1)
            .view(layer.weight.data.shape)
    )


def load_tf_basicConv2d(weights, layer):
    load_tf_conv2d(weights[0], layer.conv)
    load_tf_batchNorm(weights[1:], layer.bn)


def load_tf_linear(weights, layer):
    if isinstance(weights, list):
        if len(weights) == 2:
            layer.bias.data = (
                torch.tensor(weights[1])
                    .view(layer.bias.data.shape)
            )
        weights = weights[0]
    layer.weight.data = (
        torch.tensor(weights)
            .permute(1, 0)
            .view(layer.weight.data.shape)
    )


def load_tf_block35(weights, layer):
    load_tf_basicConv2d(weights[:4], layer.branch0)
    load_tf_basicConv2d(weights[4:8], layer.branch1[0])
    load_tf_basicConv2d(weights[8:12], layer.branch1[1])
    load_tf_basicConv2d(weights[12:16], layer.branch2[0])
    load_tf_basicConv2d(weights[16:20], layer.branch2[1])
    load_tf_basicConv2d(weights[20:24], layer.branch2[2])
    load_tf_conv2d(weights[24:26], layer.conv2d)


def load_tf_block17_8(weights, layer):
    load_tf_basicConv2d(weights[:4], layer.branch0)
    load_tf_basicConv2d(weights[4:8], layer.branch1[0])
    load_tf_basicConv2d(weights[8:12], layer.branch1[1])
    load_tf_basicConv2d(weights[12:16], layer.branch1[2])
    load_tf_conv2d(weights[16:18], layer.conv2d)


def load_tf_mixed6a(weights, layer):
    if len(weights) != 16:
        raise ValueError(f'Number of weight arrays ({len(weights)}) not equal to 16')
    load_tf_basicConv2d(weights[:4], layer.branch0)
    load_tf_basicConv2d(weights[4:8], layer.branch1[0])
    load_tf_basicConv2d(weights[8:12], layer.branch1[1])
    load_tf_basicConv2d(weights[12:16], layer.branch1[2])


def load_tf_mixed7a(weights, layer):
    if len(weights) != 28:
        raise ValueError(f'Number of weight arrays ({len(weights)}) not equal to 28')
    load_tf_basicConv2d(weights[:4], layer.branch0[0])
    load_tf_basicConv2d(weights[4:8], layer.branch0[1])
    load_tf_basicConv2d(weights[8:12], layer.branch1[0])
    load_tf_basicConv2d(weights[12:16], layer.branch1[1])
    load_tf_basicConv2d(weights[16:20], layer.branch2[0])
    load_tf_basicConv2d(weights[20:24], layer.branch2[1])
    load_tf_basicConv2d(weights[24:28], layer.branch2[2])


def load_tf_repeats(weights, layer, rptlen, subfun):
    if len(weights) % rptlen != 0:
        raise ValueError(f'Number of weight arrays ({len(weights)}) not divisible by {rptlen}')
    weights_split = [weights[i:i+rptlen] for i in range(0, len(weights), rptlen)]
    for i, w in enumerate(weights_split):
        subfun(w, getattr(layer, str(i)))


def load_tf_repeat_1(weights, layer):
    load_tf_repeats(weights, layer, 26, load_tf_block35)


def load_tf_repeat_2(weights, layer):
    load_tf_repeats(weights, layer, 18, load_tf_block17_8)


def load_tf_repeat_3(weights, layer):
    load_tf_repeats(weights, layer, 18, load_tf_block17_8)


def test_loaded_params(mdl, tf_params, tf_layers):
    tf_means = torch.stack([torch.tensor(p).mean() for p in tf_params])
    for name, param in mdl.named_parameters():
        pt_mean = param.data.mean()
        matching_inds = ((tf_means - pt_mean).abs() < 1e-8).nonzero()
        print(f'{name} equivalent to {[tf_layers[i] for i in matching_inds]}')


def compare_model_outputs(pt_mdl, sess, test_data):

    print('\nPassing test data through TF model\n')
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    feed_dict = {images_placeholder: test_data.numpy(), phase_train_placeholder: False}
    tf_output = torch.tensor(sess.run(embeddings, feed_dict=feed_dict))

    print(tf_output)

    print('\nPassing test data through PT model\n')
    pt_output = pt_mdl(test_data.permute(0, 3, 1, 2))
    print(pt_output)

    distance = (tf_output - pt_output).norm()
    print(f'\nDistance {distance}\n')


def load_tf_model_weights(mdl, layer_lookup, tf_mdl_dir):

    tf.reset_default_graph()
    with tf.Session() as sess:
        tf_layers, tf_params, tf_shapes = import_tf_params(tf_mdl_dir, sess)
        layer_info = get_layer_indices(layer_lookup, tf_layers)

        for layer_name, info in layer_info.items():
            print(f'Loading {info[0]}/* into {layer_name}')
            weights = [tf_params[i] for i in info[2]]
            layer = getattr(mdl, layer_name)
            info[1](weights, layer)

        test_loaded_params(mdl, tf_params, tf_layers)
        compare_model_outputs(mdl, sess, torch.randn(5, 160, 160, 3).detach())


def tensorflow2pytorch():

    layer_lookup = {
        'conv2d_1a': ['InceptionResnetV1/Conv2d_1a_3x3', load_tf_basicConv2d],
        'conv2d_2a': ['InceptionResnetV1/Conv2d_2a_3x3', load_tf_basicConv2d],
        'conv2d_2b': ['InceptionResnetV1/Conv2d_2b_3x3', load_tf_basicConv2d],
        'conv2d_3b': ['InceptionResnetV1/Conv2d_3b_1x1', load_tf_basicConv2d],
        'conv2d_4a': ['InceptionResnetV1/Conv2d_4a_3x3', load_tf_basicConv2d],
        'conv2d_4b': ['InceptionResnetV1/Conv2d_4b_3x3', load_tf_basicConv2d],
        'repeat_1': ['InceptionResnetV1/Repeat/block35', load_tf_repeat_1],
        'mixed_6a': ['InceptionResnetV1/Mixed_6a', load_tf_mixed6a],
        'repeat_2': ['InceptionResnetV1/Repeat_1/block17', load_tf_repeat_2],
        'mixed_7a': ['InceptionResnetV1/Mixed_7a', load_tf_mixed7a],
        'repeat_3': ['InceptionResnetV1/Repeat_2/block8', load_tf_repeat_3],
        'block8': ['InceptionResnetV1/Block8', load_tf_block17_8],
        'last_linear': ['InceptionResnetV1/Bottleneck/weights', load_tf_linear],
        'last_bn': ['InceptionResnetV1/Bottleneck/BatchNorm', load_tf_batchNorm],
        'logits': ['Logits', load_tf_linear],
    }

    mdl = InceptionResNetV1(num_classes=8631).eval()
    tf_mdl_dir = 'data/20180402-114759'
    load_tf_model_weights(mdl, layer_lookup, tf_mdl_dir)
    torch.save(mdl.state_dict(), f'{tf_mdl_dir}-vggface2.pt')

    mdl = InceptionResNetV1(num_classes=10575).eval()
    tf_mdl_dir = 'data/20180408-102900'
    load_tf_model_weights(mdl, layer_lookup, tf_mdl_dir)
    torch.save(mdl.state_dict(), f'{tf_mdl_dir}-casia-webface.pt')
