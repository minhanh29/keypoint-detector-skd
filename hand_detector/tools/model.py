from functools import wraps, partial, reduce
from tensorflow.keras.layers import MaxPooling2D, Lambda, LeakyReLU, Conv2D,\
    BatchNormalization, Concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Input
from tensorflow_model_optimization.sparsity import keras as sparsity
from tools.loss import yolo3_loss
from tools.postprocess import batched_yolo3_postprocess, batched_yolo3_prenms, Yolo3PostProcessLayer
import tensorflow as tf


L2_FACTOR = 1e-5


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def YoloConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    return Conv2D(*args, **yolo_conv_kwargs)


_DarknetConv2D = partial(YoloConv2D, padding='same')


@wraps(YoloConv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet weight regularizer for YoloConv2D."""
    darknet_conv_kwargs = kwargs
    return _DarknetConv2D(*args, **darknet_conv_kwargs)


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    return BatchNorm(*args, **kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        LeakyReLU(alpha=0.1))


def get_pruning_model(model, begin_step, end_step):
    import tensorflow as tf
    if tf.__version__.startswith('2'):
        # model pruning API is not supported in TF 2.0 yet
        raise Exception('model pruning is not fully supported in TF 2.x, Please switch env to TF 1.x for this feature')

    pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                   final_sparsity=0.7,
                                                   begin_step=begin_step,
                                                   end_step=end_step,
                                                   frequency=100)
    }

    pruning_model = sparsity.prune_low_magnitude(model, **pruning_params)
    return pruning_model


def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        model.add_metric(metric, name=name, aggregation='mean')


def tiny_yolo4_body(inputs, num_anchors, num_classes, weights_path=None):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    # return load_model('weights/yolov4-tiny.h5', custom_objects={'tf': tf})
    #feature map 2 (26x26x256 for 416 input)
    f1 = compose(
        DarknetConv2D_BN_Leaky(32, (3, 3), strides=(2, 2)),
        DarknetConv2D_BN_Leaky(64, (3, 3), strides=(2, 2)),
        DarknetConv2D_BN_Leaky(64, (3, 3)))(inputs)

    f2 = compose(
        Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[1],
               name='group_route_3'),
        DarknetConv2D_BN_Leaky(32, (3, 3))
    )(f1)

    x1 = DarknetConv2D_BN_Leaky(32, (3, 3))(f2)

    y1 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(64, (1, 1))
    )([x1, f2])

    y1_2 = compose(
        Concatenate(),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
    )([f1, y1])

    y2 = compose(
        Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[1],
               name='group_route_11'),
        DarknetConv2D_BN_Leaky(64, (3, 3))
    )(y1_2)

    x2 = DarknetConv2D_BN_Leaky(64, (3, 3))(y2)

    y3 = compose(
        Concatenate(),
    )([x2, y2])

    x3 = DarknetConv2D_BN_Leaky(128, (1, 1))(y3)

    y4 = compose(
        Concatenate(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3))
    )([y1_2, x3])

    y5 = compose(
        Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[1],
               name='group_route_19'),
        DarknetConv2D_BN_Leaky(128, (3, 3))
    )(y4)

    x4 = DarknetConv2D_BN_Leaky(128, (3, 3))(y5)

    y6 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (1, 1))
    )([x4, y5])

    y7 = compose(
        Concatenate(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3))
    )([y4, y6])

    #feature map 1 transform
    x5 = DarknetConv2D_BN_Leaky(256, (1, 1))(y7)

    #feature map 1 output (13x53 for 416 input)
    y8 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1),
                      name='predict_conv_1'))(x5)

    #upsample fpn merge for feature map 1 & 2
    x6 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x5)

    #feature map 2 output (26x66 for 416 input)
    y9 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1), name='predict_conv_2'))([x6, y6])

    return Model(inputs, [y8, y9])


def custom_tiny_yolo4_body(inputs, num_anchors, num_classes, weights_path=None):
    '''Create a custom Tiny YOLO_v3 model, use
       pre-trained weights from darknet and fit
       for our target classes.'''
    # num_classes_coco = 80
    base_model = tiny_yolo4_body(inputs, num_anchors, num_classes)

    #get conv output in original network
    # y1 = base_model.get_layer('leaky_re_lu_16').output
    # y2 = base_model.get_layer('leaky_re_lu_18').output
    y1 = base_model.layers[-4].output
    y2 = base_model.layers[-3].output
    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_1')(y1)
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1), name='predict_conv_2')(y2)
    return Model(inputs, [y1, y2])


def get_yolo4_model(num_feature_layers, num_anchors, num_classes, input_tensor=None, input_shape=None, model_pruning=False, pruning_end_step=10000):
    #prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    #Tiny YOLOv3 model has 6 anchors and 2 feature layers
    backbone_len = 50

    model_body = tiny_yolo4_body(input_tensor, num_anchors//2, num_classes)

    if model_pruning:
        model_body = get_pruning_model(model_body, begin_step=0, end_step=pruning_end_step)

    return model_body, backbone_len


def get_yolo4_train_model(anchors, num_classes, weights_path=None, freeze_level=1, optimizer=Adam(lr=1e-3, decay=0), label_smoothing=0, elim_grid_sense=False, model_pruning=False, pruning_end_step=10000):
    '''create the training model, for YOLOv3'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv3 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    y_true = [Input(shape=(None, None, 3, num_classes+5), name='y_true_{}'.format(l)) for l in range(num_feature_layers)]

    model_body, backbone_len = get_yolo4_model(num_feature_layers, num_anchors, num_classes, model_pruning=model_pruning, pruning_end_step=pruning_end_step)
    print(f'Create yolov4 tiny model with {num_anchors} anchors and {num_classes} classes.')
    print('model layer number:', len(model_body.layers))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers.
        num = (backbone_len, len(model_body.layers)-3)[freeze_level-1]
        for i in range(num):
            model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    elif freeze_level == 0:
        # Unfreeze all layers.
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable= True
        print('Unfreeze all of the layers.')

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo3_loss, name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'label_smoothing': label_smoothing, 'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    loss_dict = {'location_loss':location_loss, 'confidence_loss':confidence_loss, 'class_loss':class_loss}
    add_metrics(model, loss_dict)

    model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # use custom yolo_loss Lambda layer

    return model


def get_yolo4_inference_model(anchors, num_classes, weights_path=None, input_shape=None, confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):
    '''create the inference model, for YOLOv3'''
    #K.clear_session() # get a new session
    num_anchors = len(anchors)
    #YOLOv3 model has 9 anchors and 3 feature layers but
    #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    #so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')

    model_body, _ = get_yolo4_model(num_feature_layers, num_anchors, num_classes, input_shape=input_shape)
    print(f'Create yolov4 tiny model with {num_anchors} anchors and {num_classes} classes.')

    if weights_path:
        model_body.load_weights(weights_path)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batched_yolo3_postprocess, name='yolo3_postprocess',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'confidence': confidence, 'iou_threshold': iou_threshold, 'elim_grid_sense': elim_grid_sense})(
        [*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model

