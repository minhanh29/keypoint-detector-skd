import os
import numpy as np
import operator
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
import bokeh
import bokeh.io as bokeh_io
import bokeh.plotting as bokeh_plotting

from tools.utils import preprocess_image, postprocess_np,\
    get_custom_objects

from tools.postprocess_np import yolo3_postprocess_np

def box_iou(pred_box, gt_box):
    '''
    Calculate iou for predict box and ground truth box
    Param
         pred_box: predict box coordinate
                   (xmin,ymin,xmax,ymax) format
         gt_box: ground truth box coordinate
                 (xmin,ymin,xmax,ymax) format
    Return
         iou value
    '''
    # get intersection box
    inter_box = [max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])]
    inter_w = max(0.0, inter_box[2] - inter_box[0] + 1)
    inter_h = max(0.0, inter_box[3] - inter_box[1] + 1)

    # compute overlap (IoU) = area of intersection / area of union
    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    inter_area = inter_w * inter_h
    union_area = pred_area + gt_area - inter_area
    return 0 if union_area == 0 else float(inter_area) / float(union_area)


def annotation_parse(annotation_lines, class_names):
    '''
    parse annotation lines to get image dict and ground truth class dict

    image dict would be like:
    annotation_records = {
        '/path/to/000001.jpg': {'100,120,200,235':'dog',
        '85,63,156,128':'car', ...},
        ...
    }

    ground truth class dict would be like:
    classes_records = {
        'car': [
                ['000001.jpg','100,120,200,235'],
                ['000002.jpg','85,63,156,128'],
                ...
               ],
        ...
    }
    '''
    annotation_records = OrderedDict()
    classes_records = OrderedDict({class_name: []
                                   for class_name in class_names})

    for line in annotation_lines:
        box_records = {}
        image_name = line.split(' ')[0]
        boxes = line.split(' ')[1:]
        for box in boxes:
            # strip box coordinate and class
            class_name = class_names[int(box.split(',')[-1])]
            coordinate = ','.join(box.split(',')[:-1])
            box_records[coordinate] = class_name

            # append or add ground truth class item
            record = [os.path.basename(image_name), coordinate]
            if class_name in classes_records:
                classes_records[class_name].append(record)
            else:
                classes_records[class_name] = list([record])
        annotation_records[image_name] = box_records

    return annotation_records, classes_records


def yolo_predict_keras(model, image, anchors, num_classes, model_image_size,
                       conf_threshold, v5_decode):
    image_data = preprocess_image(image, model_image_size)
    # origin image shape, in (height, width) format
    image_shape = tuple(reversed(image.size))

    prediction = model.predict([image_data])
    pred_boxes, pred_classes, pred_scores = yolo3_postprocess_np(prediction, image_shape, anchors, num_classes, model_image_size, max_boxes=100, confidence=conf_threshold, elim_grid_sense=False)

    return pred_boxes, pred_classes, pred_scores


def transform_gt_record(gt_records, class_names):
    '''
    Transform the Ground Truth records of a image to prediction format, in
    order to show & compare in result pic.

    Ground Truth records is a dict with format:
        {'100,120,200,235':'dog', '85,63,156,128':'car', ...}

    Prediction format:
        (boxes, classes, scores)
    '''
    if gt_records is None or len(gt_records) == 0:
        return [], [], []

    gt_boxes = []
    gt_classes = []
    gt_scores = []
    for (coordinate, class_name) in gt_records.items():
        gt_box = [int(x) for x in coordinate.split(',')]
        gt_class = class_names.index(class_name)

        gt_boxes.append(gt_box)
        gt_classes.append(gt_class)
        gt_scores.append(1.0)

    return np.array(gt_boxes), np.array(gt_classes), np.array(gt_scores)


def get_prediction_class_records(model, model_format, annotation_records,
                                 anchors, class_names, model_image_size,
                                 conf_threshold, v5_decode,
                                 save_result):
    '''
    Do the predict with YOLO model on annotation images
    to get predict class dict

    predict class dict would contain image_name, coordinary and score, and
    sorted by score:
    pred_classes_records = {
        'car': [
                ['000001.jpg','94,115,203,232',0.98],
                ['000002.jpg','82,64,154,128',0.93],
                ...
               ],
        ...
    }
    '''
    # create txt file to save prediction result, with
    # save format as annotation file but adding score, like:
    #
    # path/to/img1.jpg 50,100,150,200,0,0.86 30,50,200,120,3,0.95
    #
    os.makedirs('result', exist_ok=True)
    result_file = open(os.path.join('result', 'detection_result.txt'), 'w')

    pred_classes_records = OrderedDict()
    pbar = tqdm(total=len(annotation_records), desc='Eval model')
    for (image_name, gt_records) in annotation_records.items():
        image = Image.open(image_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image, dtype='uint8')

        # support of tflite model
        pred_boxes, pred_classes, pred_scores = yolo_predict_keras(model, image, anchors, len(class_names), model_image_size, conf_threshold, v5_decode)

        pbar.update(1)

        # save prediction result to txt
        result_file.write(image_name)
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            xmin, ymin, xmax, ymax = box
            box_annotation = " %d,%d,%d,%d,%d,%f" % (
                xmin, ymin, xmax, ymax, cls, score)
            result_file.write(box_annotation)
        result_file.write('\n')
        result_file.flush()

        # if save_result:
        #     gt_boxes, gt_classes, gt_scores = transform_gt_record(gt_records,
        #                                                           class_names)

        #     result_dir = os.path.join('result', 'detection')
        #     os.makedirs(result_dir, exist_ok=True)
        #     colors = get_colors(class_names)
        #     image_array = draw_boxes(image_array, gt_boxes, gt_classes,
        #                              gt_scores, class_names, colors=None,
        #                              show_score=False)
        #     image_array = draw_boxes(image_array, pred_boxes, pred_classes,
        #                              pred_scores, class_names, colors)
        #     image = Image.fromarray(image_array)
        #     # here we handle the RGBA image
        #     if(len(image.split()) == 4):
        #         r, g, b, a = image.split()
        #         image = Image.merge("RGB", (r, g, b))
        #     image.save(os.path.join(result_dir,
        #                             image_name.split(os.path.sep)[-1]))

        # Nothing detected
        if pred_boxes is None or len(pred_boxes) == 0:
            continue

        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            pred_class_name = class_names[cls]
            xmin, ymin, xmax, ymax = box
            coordinate = "{},{},{},{}".format(xmin, ymin, xmax, ymax)

            # append or add predict class item
            record = [os.path.basename(image_name), coordinate, score]
            if pred_class_name in pred_classes_records:
                pred_classes_records[pred_class_name].append(record)
            else:
                pred_classes_records[pred_class_name] = list([record])

    # sort pred_classes_records for each class according to score
    for pred_class_list in pred_classes_records.values():
        pred_class_list.sort(key=lambda ele: ele[2], reverse=True)

    pbar.close()
    result_file.close()
    return pred_classes_records


def match_gt_box(pred_record, gt_records, iou_threshold=0.5):
    '''
    Search gt_records list and try to find a matching box for the predict box

    Param
         pred_record: with format ['image_file', 'xmin,ymin,xmax,ymax', score]
         gt_records: record list with format
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ...
                     ]
         iou_threshold:

         pred_record and gt_records should be from same annotation image file

    Return
         matching gt_record index. -1 when there's no matching gt
    '''
    max_iou = 0.0
    max_index = -1
    # get predict box coordinate
    pred_box = [float(x) for x in pred_record[1].split(',')]

    for i, gt_record in enumerate(gt_records):
        # get ground truth box coordinate
        gt_box = [float(x) for x in gt_record[1].split(',')]
        iou = box_iou(pred_box, gt_box)

        # if the ground truth has been assigned to other
        # prediction, we couldn't reuse it
        if iou > max_iou and gt_record[2] == 'unused' and\
                pred_record[0] == gt_record[0]:
            max_iou = iou
            max_index = i

    # drop the prediction if couldn't match iou threshold
    if max_iou < iou_threshold:
        max_index = -1

    return max_index


def draw_rec_prec(rec, prec, mrec, mprec, class_name, ap):
    """
     Draw plot
    """
    plt.plot(rec, prec, '-o')
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y,
                     alpha=0.2, edgecolor='r')
    fig = plt.gcf()
    fig.canvas.set_window_title('AP ' + class_name)
    plt.title('class: ' + class_name + ' AP = {}%'.format(ap*100))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    rec_prec_plot_path = os.path.join('result', 'classes')
    os.makedirs(rec_prec_plot_path, exist_ok=True)
    fig.savefig(os.path.join(rec_prec_plot_path, class_name + ".png"))
    plt.cla()


def get_rec_prec(true_positive, false_positive, gt_records):
    '''
    Calculate precision/recall based on true_positive, false_positive
    result.
    '''
    cumsum = 0
    for idx, val in enumerate(false_positive):
        false_positive[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(true_positive):
        true_positive[idx] += cumsum
        cumsum += val

    rec = true_positive[:]
    for idx, val in enumerate(true_positive):
        rec[idx] = (float(true_positive[idx]) / len(gt_records))\
            if len(gt_records) != 0 else 0

    prec = true_positive[:]
    for idx, val in enumerate(true_positive):
        prec[idx] = float(true_positive[idx]) / (false_positive[idx] +
                                                 true_positive[idx])

    return rec, prec


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def generate_rec_prec_html(mrec, mprec, scores, class_name, ap):
    """
     generate dynamic P-R curve HTML page for each class
    """
    # bypass invalid class
    if len(mrec) == 0 or len(mprec) == 0 or len(scores) == 0:
        return

    rec_prec_plot_path = os.path.join('result', 'classes')
    os.makedirs(rec_prec_plot_path, exist_ok=True)
    bokeh_io.output_file(os.path.join(rec_prec_plot_path, class_name + '.html'), title='P-R curve for ' + class_name)

    # prepare curve data
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    score_on_curve = [0.0] + scores[:-1] + [0.0] + [scores[-1]] + [1.0]
    source = bokeh.models.ColumnDataSource(data={
      'rec': area_under_curve_x,
      'prec': area_under_curve_y,
      'score': score_on_curve,
    })

    # prepare plot figure
    plt_title = 'class: ' + class_name + ' AP = {}%'.format(ap*100)
    plt = bokeh_plotting.figure(plot_height=200, plot_width=200, tools="",
                                toolbar_location=None, title=plt_title,
                                sizing_mode="scale_width")
    plt.background_fill_color = "#f5f5f5"
    plt.grid.grid_line_color = "white"
    plt.xaxis.axis_label = 'Recall'
    plt.yaxis.axis_label = 'Precision'
    plt.axis.axis_line_color = None

    # draw curve data
    plt.line(x='rec', y='prec', line_width=2, color='#ebbd5b', source=source)
    plt.add_tools(bokeh.models.HoverTool(
      tooltips=[
        ('score', '@score{0.0000 a}'),
        ('Prec', '@prec'),
        ('Recall', '@rec'),
      ],
      formatters={
        'rec': 'printf',
        'prec': 'printf',
      },
      mode='vline'
    ))
    bokeh_io.save(plt)
    return


def calc_AP(gt_records, pred_records, class_name, iou_threshold, show_result):
    '''
    Calculate AP value for one class records

    Param
         gt_records: ground truth records list for one class, with format:
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ...
                     ]
         pred_records: predict records for one class, with format
         (in score descending order):
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax', score],
                      ['image_file', 'xmin,ymin,xmax,ymax', score],
                      ...
                     ]
    Return
         AP value for the class
    '''
    # append usage flag in gt_records for matching gt search
    gt_records = [gt_record + ['unused'] for gt_record in gt_records]

    # prepare score list for generating P-R html page
    scores = [pred_record[2] for pred_record in pred_records]

    # init true_positive and false_positive list
    nd = len(pred_records)  # number of predict data
    true_positive = [0] * nd
    false_positive = [0] * nd
    true_positive_count = 0
    # assign predictions to ground truth objects
    for idx, pred_record in enumerate(pred_records):
        # filter out gt record from same image
        image_gt_records = [gt_record for gt_record in gt_records
                            if gt_record[0] == pred_record[0]]

        i = match_gt_box(pred_record, image_gt_records,
                         iou_threshold=iou_threshold)
        if i != -1:
            # find a valid gt obj to assign, set
            # true_positive list and mark image_gt_records.
            #
            # trick: gt_records will also be marked
            # as 'used', since image_gt_records is a
            # reference list
            image_gt_records[i][2] = 'used'
            true_positive[idx] = 1
            true_positive_count += 1
        else:
            false_positive[idx] = 1

    # compute precision/recall
    rec, prec = get_rec_prec(true_positive, false_positive, gt_records)
    ap, mrec, mprec = voc_ap(rec, prec)
    if show_result:
        draw_rec_prec(rec, prec, mrec, mprec, class_name, ap)
        generate_rec_prec_html(mrec, mprec, scores, class_name, ap)

    return ap, true_positive_count


def get_mean_metric(metric_records, gt_classes_records):
    '''
    Calculate mean metric, but only count classes
    which have ground truth object

    Param
        metric_records: metric dict like:
            metric_records = {
                'aeroplane': 0.79,
                'bicycle': 0.79,
                    ...
                'tvmonitor': 0.71,
            }
        gt_classes_records: ground truth class dict like:
            gt_classes_records = {
                'car': [
                    ['000001.jpg','100,120,200,235'],
                    ['000002.jpg','85,63,156,128'],
                    ...
                    ],
                ...
            }
    Return
         mean_metric: float value of mean metric
    '''
    mean_metric = 0.0
    count = 0
    for (class_name, metric) in metric_records.items():
        if (class_name in gt_classes_records) and\
                (len(gt_classes_records[class_name]) != 0):
            mean_metric += metric
            count += 1
    mean_metric = (mean_metric/count)*100 if count != 0 else 0.0
    return mean_metric


def compute_mAP_PascalVOC(annotation_records, gt_classes_records,
                          pred_classes_records, class_names, iou_threshold,
                          show_result=True):
    '''
    Compute PascalVOC style mAP
    '''
    APs = {}
    count_true_positives = {class_name: 0 for class_name in
                            list(gt_classes_records.keys())}
    # get AP value for each of the ground truth classes
    for _, class_name in enumerate(class_names):
        # if there's no gt obj for a class, record 0
        if class_name not in gt_classes_records:
            APs[class_name] = 0.
            continue
        gt_records = gt_classes_records[class_name]
        # if we didn't detect any obj for a class, record 0
        if class_name not in pred_classes_records:
            APs[class_name] = 0.
            continue
        pred_records = pred_classes_records[class_name]
        ap, true_positive_count = calc_AP(gt_records, pred_records, class_name,
                                          iou_threshold, show_result)
        APs[class_name] = ap
        count_true_positives[class_name] = true_positive_count

    # sort AP result by value, in descending order
    APs = OrderedDict(sorted(APs.items(), key=operator.itemgetter(1),
                             reverse=True))

    # get mAP percentage value
    # mAP = np.mean(list(APs.values()))*100
    mAP = get_mean_metric(APs, gt_classes_records)

    # get GroundTruth count per class
    gt_counter_per_class = {}
    for (class_name, info_list) in gt_classes_records.items():
        gt_counter_per_class[class_name] = len(info_list)

    # get Precision count per class
    pred_counter_per_class = {class_name: 0 for class_name in
                              list(gt_classes_records.keys())}
    for (class_name, info_list) in pred_classes_records.items():
        pred_counter_per_class[class_name] = len(info_list)

    # get the precision & recall
    precision_dict = {}
    recall_dict = {}
    for (class_name, gt_count) in gt_counter_per_class.items():
        if (class_name not in pred_counter_per_class) or\
                (class_name not in count_true_positives) or\
                pred_counter_per_class[class_name] == 0:
            precision_dict[class_name] = 0.
        else:
            precision_dict[class_name] = float(count_true_positives[class_name]) / pred_counter_per_class[class_name]

        if class_name not in count_true_positives or gt_count == 0:
            recall_dict[class_name] = 0.
        else:
            recall_dict[class_name] = float(count_true_positives[class_name]) / gt_count

    # get mPrec, mRec
    # mPrec = np.mean(list(precision_dict.values()))*100
    # mRec = np.mean(list(recall_dict.values()))*100
    mPrec = get_mean_metric(precision_dict, gt_classes_records)
    mRec = get_mean_metric(recall_dict, gt_classes_records)

    if show_result:
        # show result
        print('\nPascal VOC AP evaluation')
        for (class_name, AP) in APs.items():
            print('%s: AP %.4f, precision %.4f, recall %.4f' %
                  (class_name, AP, precision_dict[class_name],
                   recall_dict[class_name]))
        print('mAP@IoU=%.2f result: %f' % (iou_threshold, mAP))
        print('mPrec@IoU=%.2f result: %f' % (iou_threshold, mPrec))
        print('mRec@IoU=%.2f result: %f' % (iou_threshold, mRec))

    # return mAP percentage value
    return mAP, APs


def eval_AP(model, model_format, annotation_lines, anchors, class_names,
            model_image_size, eval_type, iou_threshold, conf_threshold,
            v5_decode, save_result):
    annotation_records, gt_classes_records = annotation_parse(annotation_lines,
                                                              class_names)
    pred_classes_records = get_prediction_class_records(model, model_format, annotation_records, anchors, class_names, model_image_size, conf_threshold, v5_decode, save_result)
    AP = 0.0

    if eval_type == 'VOC':
        AP, APs = compute_mAP_PascalVOC(annotation_records, gt_classes_records,
                                        pred_classes_records, class_names,
                                        iou_threshold)

    return AP


def load_eval_model(model_path):
    # normal keras h5 model
    custom_object_dict = get_custom_objects()

    model = load_model(model_path, compile=False,
                       custom_objects=custom_object_dict)
    model_format = 'H5'
    K.set_learning_phase(0)

    return model, model_format
