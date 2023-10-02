# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

import argparse
from pathlib import Path

import cv2
import torch

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import TestRequirements
from boxmot.utils.torch_utils import select_device

__tr = TestRequirements()
__tr.check_packages(('ultralytics==8.0.124',))  # install

from detectors import get_yolo_inferer
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.engine.model import TASK_MAP, YOLO
from ultralytics.yolo.utils import IterableSimpleNamespace, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.plotting import save_one_box
from utils import write_MOT_results

from boxmot.utils import EXAMPLES


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (opt.tracking_method + '.yaml')
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.device,
            predictor.args.half,
            predictor.args.per_class,
            predictor.args.conf_gun,
            predictor.args.overlap_gun
        )
        predictor.trackers.append(tracker)


@torch.no_grad()
def run(args):

    model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)

    # extract task predictor
    predictor = model.predictor

    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)
    predictor.args.device = select_device(args['device'])
    LOGGER.info(args)

    # setup source and model
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    predictor.setup_source(predictor.args.source)

    predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
    predictor.save_dir = increment_path(Path(predictor.args.project) /
                                        predictor.args.name, exist_ok=predictor.args.exist_ok)

    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt
         else predictor.save_dir).mkdir(parents=True, exist_ok=True)

    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(
            imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz)
        )
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = (
        0,
        [],
        None,
        (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    )
    predictor.add_callback('on_predict_start', on_predict_start)
    predictor.run_callbacks('on_predict_start')

    # get yolo class based on model name
    Yolo = get_yolo_inferer(args['yolo_model'])
    # initialize class
    model = Yolo(
        model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
        device=predictor.device,
        args=predictor.args
    )

    for frame_idx, batch in enumerate(predictor.dataset):
        predictor.run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch

        n = len(im0s)
        predictor.results = [None] * n

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model.inference(im=im)

        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks('on_predict_postprocess_end')

        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):

            if predictor.dataset.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)

            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                # get tracker predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach().numpy(), im0)
            predictor.results[i].speed = {
                'preprocess': predictor.profilers[0].dt * 1E3 / n,
                'inference': predictor.profilers[1].dt * 1E3 / n,
                'postprocess': predictor.profilers[2].dt * 1E3 / n,
                'tracking': predictor.profilers[3].dt * 1E3 / n
            }

            # filter boxes masks and pose results by tracking results
            model.filter_results(i, predictor)
            # overwrite bbox results with tracker predictions
            model.overwrite_results(i, im0.shape[:2], predictor)

            # write inference results to a file or directory
            if (predictor.args.verbose or predictor.args.save or
               predictor.args.save_txt or predictor.args.show or
               predictor.args.save_id_crops):

                s += predictor.write_results(i, predictor.results, (p, im, im0))
                predictor.txt_path = Path(predictor.txt_path)

                # write MOT specific results
                if predictor.args.source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                # mot txt called the same as the parent name to perform inference on
                elif 'MOT16' or 'MOT17' or 'MOT20' in predictor.args.source:
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.parent.name
                # mot txt called the same as the parent name to perform inference on
                else:

                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name

                if predictor.tracker_outputs[i].size != 0 and predictor.args.save_mot:
                    write_MOT_results(
                        predictor.MOT_txt_path,
                        predictor.results[i],
                        frame_idx,
                        i,
                    )

                if predictor.args.save_id_crops:
                    for d in predictor.results[i].boxes:
                        save_one_box(
                            d.xyxy,
                            im0.copy(),
                            file=(predictor.save_dir / 'crops' /
                                  str(int(d.cls.cpu().numpy().item())) /
                                  str(int(d.id.cpu().numpy().item())) / f'{p.stem}.jpg'),
                            BGR=True
                        )

            # display an image in a window using OpenCV imshow()
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p.parent)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

        predictor.run_callbacks('on_predict_batch_end')

        # print time (inference-only)
        if predictor.args.verbose:
            s_t = f'YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms'
            LOGGER.info(f'{s}{s_t}')

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
        LOGGER.info(f'Speed: %.1fms preproc, %.1fms inference, %.1fms postproc, %.1fms tracking per image at shape '
                    f'{(1, 3, *predictor.args.imgsz)}' % t)
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks('on_predict_end')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_market1501.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, deepocsort_confirm, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=EXAMPLES / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='hide labels when show')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--conf_gun', type=float, default=0.35,
                        help='confidence threshold for gun detection-based shooter confirmation')
    parser.add_argument('--overlap_gun', type=float, default=0.75,
                        help='Overlap threshold for gun detection-based shooter confirmation')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
