import copy
import tensorflow as tf
from scipy.special import expit, softmax
import numpy as np
import random
import time
import cv2
import colorsys
from PIL import Image
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay,\
    PolynomialDecay, PiecewiseConstantDecay
from tensorflow.keras.experimental import CosineDecay


def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,
    }

    return custom_objects_dict


def optimize_tf_gpu(tf, K):
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        session = tf.Session(config=config)

        # set session
        K.set_session(session)


def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def get_dataset(annotation_file, shuffle=True):
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        np.random.seed(int(time.time()))
        np.random.shuffle(lines)

    return lines


def my_letterbox_resize(img, target_size):
    src_h = img.shape[0]
    src_w = img.shape[1]
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)

    img = cv2.resize(img, (padding_w, padding_h))

    off_x = abs(padding_w - target_w)
    off_y = abs(padding_h - target_h)
    left = int(off_x / 2)
    top = int(off_y / 2)
    bottom = off_y - top
    right = off_x - left

    # black padding
    newImg = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(128, 128, 128))

    return newImg


def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    return image

def my_preprocess_image(image, model_image_size):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = my_letterbox_resize(image, tuple(reversed(model_image_size)))
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data


def preprocess_image(image, model_image_size):
    """
    Prepare model input image data with letterbox
    resize, normalize and dim expansion

    # Arguments
        image: origin input image
            PIL Image object containing image data
        model_image_size: model input image size
            tuple of format (height, width).

    # Returns
        image_data: numpy array of image data for model input.
    """
    resized_image = letterbox_resize(image, tuple(reversed(model_image_size)))
    image_data = np.asarray(resized_image).astype('float32')
    image_data = normalize_image(image_data)
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image_data


def yolo_decode(prediction, anchors, num_classes, input_dims, scale_x_y=None, use_softmax=False):
    '''Decode final layer features to bounding box parameters.'''
    batch_size = np.shape(prediction)[0]
    num_anchors = len(anchors)

    grid_size = np.shape(prediction)[1:3]

    prediction = np.reshape(prediction, (batch_size, grid_size[0] * grid_size[1] * num_anchors, num_classes + 5))

    ################################
    # generate x_y_offset grid map
    grid_y = np.arange(grid_size[0])
    grid_x = np.arange(grid_size[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    ################################

    # Log space transform of the height and width
    anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
    anchors = np.expand_dims(anchors, 0)
    box_xy = (expit(prediction[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
    box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(input_dims)[::-1]

    # Sigmoid objectness scores
    objectness = expit(prediction[..., 4])
    objectness = np.expand_dims(objectness, -1)

    if use_softmax:
        # Softmax class scores
        class_scores = softmax(prediction[..., 5:], axis=-1)
    else:
        # Sigmoid class scores
        class_scores = expit(prediction[..., 5:])

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo_correct_boxes(predictions, img_shape, model_image_size):
    '''rescale predicition boxes back to original image shape'''
    box_xy = predictions[..., :2]
    box_wh = predictions[..., 2:4]
    objectness = np.expand_dims(predictions[..., 4], -1)
    class_scores = predictions[..., 5:]

    # model_image_size & image_shape should be (height, width) format
    model_image_size = np.array(model_image_size, dtype='float32')
    image_shape = np.array(img_shape, dtype='float32')
    height, width = image_shape

    new_shape = np.round(image_shape * np.min(model_image_size/image_shape))
    offset = (model_image_size-new_shape)/2./model_image_size
    scale = model_image_size/new_shape
    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    scale = scale[..., ::-1]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    box_xy *= image_wh
    box_wh *= image_wh

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def yolo_handle_predictions(predictions, image_shape, max_boxes=100,
                            confidence=0.1, iou_threshold=0.4, ):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    # filter boxes with confidence threshold
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    # Boxes, Classes and Scores returned from NMS
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores,
                                             iou_threshold,
                                             confidence=confidence)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes).astype('int32')
        scores = np.concatenate(n_scores)
        boxes, classes, scores = filter_boxes(boxes, classes,
                                              scores, max_boxes)

        return boxes, classes, scores

    else:
        return [], [], []


def filter_boxes(boxes, classes, scores, max_boxes):
    '''
    Sort the prediction boxes according to score
    and only pick top "max_boxes" ones
    '''
    # sort result according to scores
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    nboxes = boxes[sorted_indices]
    nclasses = classes[sorted_indices]
    nscores = scores[sorted_indices]

    # only pick max_boxes
    nboxes = nboxes[:max_boxes]
    nclasses = nclasses[:max_boxes]
    nscores = nscores[:max_boxes]

    return nboxes, nclasses, nscores


def box_iou(boxes):
    """
    Calculate IoU value of 1st box with other boxes of a box array

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    iou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[1:], x[0])
    inter_ymin = np.maximum(y[1:], y[0])
    inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
    inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)
    return iou


def nms_boxes(boxes, classes, scores, iou_threshold, confidence=0.1,
              is_soft=False, use_exp=False, sigma=0.5):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))

            # swap the max line and first line
            b_nms[[i, 0], :] = b_nms[[0, i], :]
            c_nms[[i, 0]] = c_nms[[0, i]]
            s_nms[[i, 0]] = s_nms[[0, i]]

            iou = box_iou(b_nms)

            # drop the last line since it has been record
            b_nms = b_nms[1:]
            c_nms = c_nms[1:]
            s_nms = s_nms[1:]

            if is_soft:
                # Soft-NMS
                if use_exp:
                    # score refresh formula:
                    # score = score * exp(-(iou^2)/sigma)
                    s_nms = s_nms * np.exp(-(iou * iou) / sigma)
                else:
                    # score refresh formula:
                    # score = score * (1 - iou) if iou > threshold
                    depress_mask = np.where(iou > iou_threshold)[0]
                    s_nms[depress_mask] = s_nms[depress_mask]*(1-iou[depress_mask])
                keep_mask = np.where(s_nms >= confidence)[0]
            else:
                # normal Hard-NMS
                keep_mask = np.where(iou <= iou_threshold)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    return nboxes, nclasses, nscores


def yolo_adjust_boxes(boxes, img_shape):
    '''
    change box format from (x,y,w,h) top left coordinate to
    (xmin,ymin,xmax,ymax) format
    '''
    if boxes is None or len(boxes) == 0:
        return []

    image_shape = np.array(img_shape, dtype='float32')
    height, width = image_shape

    adjusted_boxes = []
    for box in boxes:
        x, y, w, h = box

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
        xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
        adjusted_boxes.append([xmin, ymin, xmax, ymax])

    return np.array(adjusted_boxes, dtype=np.int32)


def postprocess_np(yolo_outputs, image_shape, anchors, num_classes,
                   model_image_size, max_boxes=100, confidence=0.1,
                   iou_threshold=0.4):

    scale_x_y = None
    predictions = yolo_decode(yolo_outputs, anchors, num_classes,
                              input_dims=model_image_size,
                              scale_x_y=scale_x_y, use_softmax=True)
    predictions = yolo_correct_boxes(predictions, image_shape,
                                     model_image_size)

    boxes, classes, scores = yolo_handle_predictions(predictions,
                                                     image_shape,
                                                     max_boxes=max_boxes,
                                                     confidence=confidence,
                                                     iou_threshold=iou_threshold)

    boxes = yolo_adjust_boxes(boxes, image_shape)

    return boxes, classes, scores


def get_lr_scheduler(learning_rate, decay_type, decay_steps):
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type is None:
        lr_scheduler = learning_rate
    elif decay_type == 'cosine':
        lr_scheduler = CosineDecay(initial_learning_rate=learning_rate,
                                   decay_steps=decay_steps, alpha=0.2)
    elif decay_type == 'exponential':
        lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate,
                                        decay_steps=decay_steps,
                                        decay_rate=0.9)
    elif decay_type == 'polynomial':
        lr_scheduler = PolynomialDecay(initial_learning_rate=learning_rate,
                                       decay_steps=decay_steps,
                                       end_learning_rate=learning_rate/100)
    elif decay_type == 'piecewise_constant':
        boundaries = [500, int(decay_steps*0.9), decay_steps]
        values = [0.001, learning_rate, learning_rate/10., learning_rate/100.]
        lr_scheduler = PiecewiseConstantDecay(boundaries=boundaries,
                                              values=values)
    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer(optim_type, learning_rate, decay_type='cosine',
                  decay_steps=100000):
    optim_type = optim_type.lower()

    lr_scheduler = get_lr_scheduler(learning_rate, decay_type, decay_steps)

    if optim_type == 'adam':
        optimizer = Adam(learning_rate=lr_scheduler, amsgrad=False)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_scheduler, rho=0.9,
                            momentum=0.0, centered=False)
    elif optim_type == 'sgd':
        optimizer = SGD(learning_rate=lr_scheduler,
                        momentum=0.0, nesterov=False)
    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer


def reshape_boxes(boxes, src_shape, target_shape, padding_shape, offset, horizontal_flip=False, vertical_flip=False):
    if len(boxes) > 0:
        src_w, src_h = src_shape
        target_w, target_h = target_shape
        padding_w, padding_h = padding_shape
        dx, dy = offset

        # shuffle and reshape boxes
        np.random.shuffle(boxes)
        boxes[:, [0,2]] = boxes[:, [0,2]]*padding_w/src_w + dx
        boxes[:, [1,3]] = boxes[:, [1,3]]*padding_h/src_h + dy
        # horizontal flip boxes if needed
        if horizontal_flip:
            boxes[:, [0,2]] = target_w - boxes[:, [2,0]]
        # vertical flip boxes if needed
        if vertical_flip:
            boxes[:, [1,3]] = target_h - boxes[:, [3,1]]

        # check box coordinate range
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > target_w] = target_w
        boxes[:, 3][boxes[:, 3] > target_h] = target_h

        # check box width and height to discard invalid box
        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(boxes_w>1, boxes_h>1)] # discard invalid box

    return boxes


def letterbox_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding

    # Arguments
        image: cv2 image
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    # Returns
        new_image: resized PIL Image object.

        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_h, src_w = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w)//2
    dy = (target_h - padding_h)//2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128,128,128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image


def get_ground_truth_data(annotation_line, input_shape,
                          augment=False, max_boxes=100):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    image_size = image.size
    model_input_size = tuple(reversed(input_shape))
    boxes = np.array([np.array(list(map(int, box.split(','))))
                      for box in line[1:]])

    # if not augment:
    new_image, padding_size, offset = letterbox_resize(image, target_size=model_input_size, return_padding_info=True)
    image_data = np.array(new_image)
    image_data = normalize_image(image_data)

    # reshape boxes
    boxes = reshape_boxes(boxes, src_shape=image_size,
                          target_shape=model_input_size,
                          padding_shape=padding_size, offset=offset)
    if len(boxes) > max_boxes:
        boxes = boxes[:max_boxes]

    # fill in box data
    box_data = np.zeros((max_boxes, 5))
    if len(boxes) > 0:
        box_data[:len(boxes)] = boxes

    return image_data, box_data

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, multi_anchor_assign, iou_thresh=0.2):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    multi_anchor_assign: boolean, whether to use iou_thresh to assign multiple
                         anchors for a single ground truth

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    #Transform box info to (x_center, y_center, box_width, box_height, cls_id)
    #and image relative coordinate.
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch_size):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue

        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Sort anchors according to IoU score
        # to find out best assignment
        best_anchors = np.argsort(iou, axis=-1)[..., ::-1]

        if not multi_anchor_assign:
            best_anchors = best_anchors[..., 0]
            # keep index dim for the loop in following
            best_anchors = np.expand_dims(best_anchors, -1)

        for t, row in enumerate(best_anchors):
            for l in range(num_layers):
                for n in row:
                    # use different matching policy for single & multi anchor assign
                    if multi_anchor_assign:
                        matching_rule = (iou[t, n] > iou_thresh and n in anchor_mask[l])
                    else:
                        matching_rule = (n in anchor_mask[l])

                    if matching_rule:
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[b, t, 4].astype('int32')
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def get_y_true_data(box_data, anchors, input_shape,
                    num_classes, multi_anchor_assign):
    '''
    Precompute y_true feature map data on a batch for training.
    y_true feature map array gives the regression targets for the ground truth
    box with shape [conv_height, conv_width, num_anchors, 6]
    '''
    y_true_data = [0 for i in range(len(box_data))]
    for i, boxes in enumerate(box_data):
        y_true_data[i] = preprocess_true_boxes(boxes, anchors, input_shape, num_classes, multi_anchor_assign)

    return np.array(y_true_data)


def yolo4_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    # prepare multiscale config
    rescale_step = 0
    input_shape_list = [(320,320), (352,352), (384,384), (416,416), (448,448), (480,480), (512,512), (544,544), (576,576), (608,608)]
    while True:
        if rescale_interval > 0:
            # Do multi-scale training on different input shape
            rescale_step = (rescale_step + 1) % rescale_interval
            if rescale_step == 0:
                input_shape = input_shape_list[random.randint(0, len(input_shape_list)-1)]

        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_ground_truth_data(annotation_lines[i], input_shape, augment=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, multi_anchor_assign)
        yield [image_data, *y_true], np.zeros(batch_size)


def yolo4_data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment=None, rescale_interval=-1, multi_anchor_assign=False, **kwargs):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return yolo4_data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign)
