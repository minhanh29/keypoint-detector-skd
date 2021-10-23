import time
import os
from tools.evaluation import eval_AP, load_eval_model
from tools.utils import get_classes, get_anchors, get_dataset


def main():
    anchors_path = os.path.join('config', 'yolov4-tiny-anchors.txt')
    classes_path = os.path.join('config', 'hand_class.txt')
    annotation_file = os.path.join('anno', 'testval.txt')
    model_path = os.path.join('model', 'model.h5')
    model_image_size = (416, 416)

    # param parse
    anchors = get_anchors(anchors_path)
    class_names = get_classes(classes_path)
    iou_threshold = 0.5
    conf_threshold = 0.2
    save_result = False

    annotation_lines = get_dataset(annotation_file, shuffle=False)
    model, model_format = load_eval_model(model_path)

    start = time.time()
    eval_AP(model, model_format, annotation_lines, anchors, class_names,
            model_image_size, 'VOC', iou_threshold, conf_threshold,
            False, save_result)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()


