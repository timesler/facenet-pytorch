import tensorflow as tf
import torch
import json
import os, sys

from dependencies.facenet.src import facenet
from dependencies.facenet.src.models import inception_resnet_v1 as tf_mdl
from dependencies.facenet.src.align import detect_face

from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import PNet, RNet, ONet


def import_tf_params(tf_mdl_dir, sess):
    """Import tensorflow model from save directory.
    
    Arguments:
        tf_mdl_dir {str} -- Location of protobuf, checkpoint, meta files.
        sess {tensorflow.Session} -- Tensorflow session object.
    
    Returns:
        (list, list, list) -- Tuple of lists containing the layer names,
            parameter arrays as numpy ndarrays, parameter shapes.
    """
    print('\nLoading tensorflow model\n')
    if callable(tf_mdl_dir):
        tf_mdl_dir(sess)
    else:
        facenet.load_model(tf_mdl_dir)

    print('\nGetting model weights\n')
    tf_layers = tf.trainable_variables()
    tf_params = sess.run(tf_layers)

    tf_shapes = [p.shape for p in tf_params]
    tf_layers = [l.name for l in tf_layers]

    if not callable(tf_mdl_dir):
        path = os.path.join(tf_mdl_dir, 'layer_description.json')
    else:
        path = 'data/layer_description.json'
    with open(path, 'w') as f:
        json.dump({l: s for l, s in zip(tf_layers, tf_shapes)}, f)

    return tf_layers, tf_params, tf_shapes


def get_layer_indices(layer_lookup, tf_layers):
    """Giving a lookup of model layer attribute names and tensorflow variable names,
    find matching parameters.
    
    Arguments:
        layer_lookup {dict} -- Dictionary mapping pytorch attribute names to (partial)
            tensorflow variable names. Expects dict of the form {'attr': ['tf_name', ...]}
            where the '...'s are ignored.
        tf_layers {list} -- List of tensorflow variable names.
    
    Returns:
        list -- The input dictionary with the list of matching inds appended to each item.
    """
    layer_inds = {}
    for name, value in layer_lookup.items():
        layer_inds[name] = value + [[i for i, n in enumerate(tf_layers) if value[0] in n]]
    return layer_inds


def load_tf_batchNorm(weights, layer):
    """Load tensorflow weights into nn.BatchNorm object.
    
    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- nn.BatchNorm.
    """
    layer.bias.data = torch.tensor(weights[0]).view(layer.bias.data.shape)
    layer.weight.data = torch.ones_like(layer.weight.data)
    layer.running_mean = torch.tensor(weights[1]).view(layer.running_mean.shape)
    layer.running_var = torch.tensor(weights[2]).view(layer.running_var.shape)


def load_tf_conv2d(weights, layer, transpose=False):
    """Load tensorflow weights into nn.Conv2d object.
    
    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- nn.Conv2d.
    """
    if isinstance(weights, list):
        if len(weights) == 2:
            layer.bias.data = (
                torch.tensor(weights[1])
                    .view(layer.bias.data.shape)
            )
        weights = weights[0]
    
    if transpose:
        dim_order = (3, 2, 1, 0)
    else:
        dim_order = (3, 2, 0, 1)

    layer.weight.data = (
        torch.tensor(weights)
            .permute(dim_order)
            .view(layer.weight.data.shape)
    )


def load_tf_conv2d_trans(weights, layer):
    return load_tf_conv2d(weights, layer, transpose=True)


def load_tf_basicConv2d(weights, layer):
    """Load tensorflow weights into grouped Conv2d+BatchNorm object.
    
    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- Object containing Conv2d+BatchNorm.
    """
    load_tf_conv2d(weights[0], layer.conv)
    load_tf_batchNorm(weights[1:], layer.bn)


def load_tf_linear(weights, layer):
    """Load tensorflow weights into nn.Linear object.
    
    Arguments:
        weights {list} -- Tensorflow parameters.
        layer {torch.nn.Module} -- nn.Linear.
    """
    if isinstance(weights, list):
        if len(weights) == 2:
            layer.bias.data = (
                torch.tensor(weights[1])
                    .view(layer.bias.data.shape)
            )
        weights = weights[0]
    layer.weight.data = (
        torch.tensor(weights)
            .transpose(-1, 0)
            .view(layer.weight.data.shape)
    )


# High-level parameter-loading functions:

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
    """Check each parameter in a pytorch model for an equivalent parameter
    in a list of tensorflow variables.
    
    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        tf_params {list} -- List of ndarrays representing tensorflow variables.
        tf_layers {list} -- Corresponding list of tensorflow variable names.
    """
    tf_means = torch.stack([torch.tensor(p).mean() for p in tf_params])
    for name, param in mdl.named_parameters():
        pt_mean = param.data.mean()
        matching_inds = ((tf_means - pt_mean).abs() < 1e-8).nonzero()
        print(f'{name} equivalent to {[tf_layers[i] for i in matching_inds]}')


def compare_model_outputs(pt_mdl, sess, test_data):
    """Given some testing data, compare the output of pytorch and tensorflow models.
    
    Arguments:
        pt_mdl {torch.nn.Module} -- Pytorch model.
        sess {tensorflow.Session} -- Tensorflow session object.
        test_data {torch.Tensor} -- Pytorch tensor.
    """
    print('\nPassing test data through TF model\n')
    if isinstance(sess, tf.Session):
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        feed_dict = {images_placeholder: test_data.numpy(), phase_train_placeholder: False}
        tf_output = torch.tensor(sess.run(embeddings, feed_dict=feed_dict))
    else:
        tf_output = sess(test_data)

    print(tf_output)

    print('\nPassing test data through PT model\n')
    pt_output = pt_mdl(test_data.permute(0, 3, 1, 2))
    print(pt_output)

    distance = (tf_output - pt_output).norm()
    print(f'\nDistance {distance}\n')


def compare_mtcnn(pt_mdl, tf_fun, sess, ind, test_data):
    tf_mdls = tf_fun(sess)
    tf_mdl = tf_mdls[ind]

    print('\nPassing test data through TF model\n')
    tf_output = tf_mdl(test_data.numpy())
    tf_output = [torch.tensor(out) for out in tf_output]
    print('\n'.join([str(o.view(-1)[:10]) for o in tf_output]))

    print('\nPassing test data through PT model\n')
    with torch.no_grad():
        pt_output = pt_mdl(test_data.permute(0, 3, 2, 1))
    pt_output = [torch.tensor(out) for out in pt_output]
    for i in range(len(pt_output)):
        if len(pt_output[i].shape) == 4:
            pt_output[i] = pt_output[i].permute(0, 3, 2, 1).contiguous()
    print('\n'.join([str(o.view(-1)[:10]) for o in pt_output]))

    distance = [(tf_o - pt_o).norm() for tf_o, pt_o in zip(tf_output, pt_output)]
    print(f'\nDistance {distance}\n')


def load_tf_model_weights(mdl, layer_lookup, tf_mdl_dir, is_resnet=True, arg_num=None):
    """Load tensorflow parameters into a pytorch model.
    
    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        layer_lookup {[type]} -- Dictionary mapping pytorch attribute names to (partial)
            tensorflow variable names, and a function suitable for loading weights.
            Expects dict of the form {'attr': ['tf_name', function]}. 
        tf_mdl_dir {str} -- Location of protobuf, checkpoint, meta files.
    """
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

        if is_resnet:
            compare_model_outputs(mdl, sess, torch.randn(5, 160, 160, 3).detach())


def tensorflow2pytorch():
    lookup_inception_resnet_v1 = {
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

    print('\nLoad VGGFace2-trained weights and save\n')
    mdl = InceptionResnetV1(num_classes=8631).eval()
    tf_mdl_dir = 'data/20180402-114759'
    data_name = 'vggface2'
    load_tf_model_weights(mdl, lookup_inception_resnet_v1, tf_mdl_dir)
    state_dict = mdl.state_dict()
    torch.save(state_dict, f'{tf_mdl_dir}-{data_name}.pt')    
    torch.save(
        {
            'logits.weight': state_dict['logits.weight'],
            'logits.bias': state_dict['logits.bias'],
        },
        f'{tf_mdl_dir}-{data_name}-logits.pt'
    )
    state_dict.pop('logits.weight')
    state_dict.pop('logits.bias')
    torch.save(state_dict, f'{tf_mdl_dir}-{data_name}-features.pt')
    
    print('\nLoad CASIA-Webface-trained weights and save\n')
    mdl = InceptionResnetV1(num_classes=10575).eval()
    tf_mdl_dir = 'data/20180408-102900'
    data_name = 'casia-webface'
    load_tf_model_weights(mdl, lookup_inception_resnet_v1, tf_mdl_dir)
    state_dict = mdl.state_dict()
    torch.save(state_dict, f'{tf_mdl_dir}-{data_name}.pt')    
    torch.save(
        {
            'logits.weight': state_dict['logits.weight'],
            'logits.bias': state_dict['logits.bias'],
        },
        f'{tf_mdl_dir}-{data_name}-logits.pt'
    )
    state_dict.pop('logits.weight')
    state_dict.pop('logits.bias')
    torch.save(state_dict, f'{tf_mdl_dir}-{data_name}-features.pt')
    
    lookup_pnet = {
        'conv1': ['pnet/conv1', load_tf_conv2d_trans],
        'prelu1': ['pnet/PReLU1', load_tf_linear],
        'conv2': ['pnet/conv2', load_tf_conv2d_trans],
        'prelu2': ['pnet/PReLU2', load_tf_linear],
        'conv3': ['pnet/conv3', load_tf_conv2d_trans],
        'prelu3': ['pnet/PReLU3', load_tf_linear],
        'conv4_1': ['pnet/conv4-1', load_tf_conv2d_trans],
        'conv4_2': ['pnet/conv4-2', load_tf_conv2d_trans],
    }
    lookup_rnet = {
        'conv1': ['rnet/conv1', load_tf_conv2d_trans],
        'prelu1': ['rnet/prelu1', load_tf_linear],
        'conv2': ['rnet/conv2', load_tf_conv2d_trans],
        'prelu2': ['rnet/prelu2', load_tf_linear],
        'conv3': ['rnet/conv3', load_tf_conv2d_trans],
        'prelu3': ['rnet/prelu3', load_tf_linear],
        'dense4': ['rnet/conv4', load_tf_linear],
        'prelu4': ['rnet/prelu4', load_tf_linear],
        'dense5_1': ['rnet/conv5-1', load_tf_linear],
        'dense5_2': ['rnet/conv5-2', load_tf_linear],
    }
    lookup_onet = {
        'conv1': ['onet/conv1', load_tf_conv2d_trans],
        'prelu1': ['onet/prelu1', load_tf_linear],
        'conv2': ['onet/conv2', load_tf_conv2d_trans],
        'prelu2': ['onet/prelu2', load_tf_linear],
        'conv3': ['onet/conv3', load_tf_conv2d_trans],
        'prelu3': ['onet/prelu3', load_tf_linear],
        'conv4': ['onet/conv4', load_tf_conv2d_trans],
        'prelu4': ['onet/prelu4', load_tf_linear],
        'dense5': ['onet/conv5', load_tf_linear],
        'prelu5': ['onet/prelu5', load_tf_linear],
        'dense6_1': ['onet/conv6-1', load_tf_linear],
        'dense6_2': ['onet/conv6-2', load_tf_linear],
        'dense6_3': ['onet/conv6-3', load_tf_linear],
    }

    print('\nLoad PNet weights and save\n')
    tf_mdl_dir = lambda sess: detect_face.create_mtcnn(sess, None)
    mdl = PNet()
    data_name = 'pnet'
    load_tf_model_weights(mdl, lookup_pnet, tf_mdl_dir, is_resnet=False, arg_num=0)
    torch.save(mdl.state_dict(), f'data/{data_name}.pt')
    tf.reset_default_graph()
    with tf.Session() as sess:
        compare_mtcnn(mdl, tf_mdl_dir, sess, 0, torch.randn(1, 256, 256, 3).detach())

    print('\nLoad RNet weights and save\n')
    mdl = RNet()
    data_name = 'rnet'
    load_tf_model_weights(mdl, lookup_rnet, tf_mdl_dir, is_resnet=False, arg_num=1)
    torch.save(mdl.state_dict(), f'data/{data_name}.pt')
    tf.reset_default_graph()
    with tf.Session() as sess:
        compare_mtcnn(mdl, tf_mdl_dir, sess, 1, torch.randn(1, 24, 24, 3).detach())

    print('\nLoad ONet weights and save\n')
    mdl = ONet()
    data_name = 'onet'
    load_tf_model_weights(mdl, lookup_onet, tf_mdl_dir, is_resnet=False, arg_num=2)
    torch.save(mdl.state_dict(), f'data/{data_name}.pt')
    tf.reset_default_graph()
    with tf.Session() as sess:
        compare_mtcnn(mdl, tf_mdl_dir, sess, 2, torch.randn(1, 48, 48, 3).detach())
