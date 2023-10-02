import numpy as np
import matplotlib.pyplot as plt

def read_bbox_file(file_path, delimiter=','):
    # Read bounding box data from a file and return it as a dictionary.
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(delimiter)
            frame_id, track_id, x, y, width, height, _, _, _ = map(float, parts)
            if frame_id not in data:
                data[frame_id] = {}
            data[frame_id][int(track_id)] = (x, y, width, height)
    return data

def evaluate_tracking(gt_data, results_data, window_size, iou_threshold):
    # Determine number of TPs, FPs, and FNs for a given window of frames
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for frame_id, gt_tracks in gt_data.items():
        for _, gt_bbox in gt_tracks.items():
            min_frame = max(frame_id - window_size, 1)
            max_frame = frame_id + window_size

            detected = False
            for i in range(int(min_frame), int(max_frame) + 1):
                if i in results_data:
                    for _, results_bbox in results_data[i].items():
                        iou = calculate_iou(gt_bbox, results_bbox)
                        if iou >= iou_threshold:
                            detected = True
                            break
                if detected:
                    break

            if detected:
                true_positives += 1
            else:
                false_negatives += 1

    for frame_id, results_tracks in results_data.items():
        for _, results_bbox in results_tracks.items():
            min_frame = max(frame_id - window_size, 1)
            max_frame = frame_id + window_size

            detected = False
            for i in range(int(min_frame), int(max_frame) + 1):
                if i in gt_data:
                    for _, gt_bbox in gt_data[i].items():
                        iou = calculate_iou(gt_bbox, results_bbox)
                        if iou >= iou_threshold:
                            detected = True
                            break
                if detected:
                    break

            if not detected:
                false_positives += 1

    return true_positives, false_positives, false_negatives


def calculate_iou(bbox1, bbox2):
    # Calculate Intersection over Union (IoU) between two bounding boxes.
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = x_intersection * y_intersection

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0  # Handle division by zero
    return iou

if __name__ == "__main__":
    # List of video numbers
    vid_nums = [1, 2, 3, 4, 5, 6]

    # Configuration parameters
    # confs_gun = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    confs_gun = [0.8]
    # confs = [0.6]
    window_sizes = list(range(1, 61))

    # Initialize lists to store precision and recall values for different configurations
    precision_values = []
    recall_values = []
    f1_score_values = []

    best_configs = []  # To store the best configurations
    best_f1_scores = []  # To store the corresponding best F1 scores

    for conf_gun in confs_gun:
        for conf in confs:
            precision_curve = []  # List to store precision values for this configuration
            recall_curve = []  # List to store recall values for this configuration
            for window_size in window_sizes:
                total_tp = 0
                total_fp = 0
                total_fn = 0
                for vid_num in vid_nums:
                    # Read ground truth data and tracking results data for each video
                    gt_path = f'MOT-ASTERS/train/MOT-ASTERS-0{vid_num}/gt/gt.txt'
                    gt_data = read_bbox_file(gt_path, delimiter=',')

                    # res_path = f'MOT-ASTERS/test_shooter0/exp_yolov8n_AugCTextured_CMasked_Real_S.pt_conf_gun{conf_gun}_conf{conf}/MOT-ASTERS-0{vid_num}.txt'
                    res_path = f'MOT-ASTERS/test_shooter0/exp_yolov8n_AugCTextured_CMasked_Real_S.pt_conf{conf}/MOT-ASTERS-0{vid_num}.txt'
                    results_data = read_bbox_file(res_path, delimiter=' ')

                    # Set the IoU threshold to consider a detection/track as a success.
                    iou_threshold = 0.5  # You can adjust this threshold as needed.

                    # Evaluate tracking results with the current window size
                    tp, fp, fn = evaluate_tracking(gt_data, results_data, window_size, iou_threshold)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0


                # Append the results to the lists
                precision_curve.append(overall_precision)
                recall_curve.append(overall_recall)

            # Calculate F1 score for the current configuration
            f1_score = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_curve, recall_curve)]
                # f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

            # Append the precision and recall curves for this configuration
            precision_values.append(precision_curve)
            recall_values.append(recall_curve)
            f1_score_values.append(f1_score)

            # Append the F1 score and configuration
            best_configs.append((conf_gun, conf))
            best_f1_scores.append(max(f1_score))

    precision_values_conf = []
    recall_values_conf = []
    f1_score_values_conf = []

    precision_curve = []  # List to store precision values for this configuration
    recall_curve = []

    for window_size in window_sizes:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for vid_num in vid_nums:
            # Read ground truth data and tracking results data for each video
            gt_path = f'MOT-ASTERS/train/MOT-ASTERS-0{vid_num}/gt/gt.txt'
            gt_data = read_bbox_file(gt_path, delimiter=',')

            # res_path = f'MOT-ASTERS/test_shooter0/exp_yolov8n_AugCTextured_CMasked_Real_S.pt_conf_gun{conf_gun}_conf{conf}/MOT-ASTERS-0{vid_num}.txt'
            res_path = f'MOT-ASTERS/test_conf/exp_yolov8n_AugCTextured_CMasked_Real_S.pt_conf_gun0.8_conf0.6/MOT-ASTERS-0{vid_num}.txt'
            results_data = read_bbox_file(res_path, delimiter=' ')

            # Set the IoU threshold to consider a detection/track as a success.
            iou_threshold = 0.5  # You can adjust this threshold as needed.

            # Evaluate tracking results with the current window size
            tp, fp, fn = evaluate_tracking(gt_data, results_data, window_size, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0


        # Append the results to the lists
        precision_curve.append(overall_precision)
        recall_curve.append(overall_recall)

    # Calculate F1 score for the current configuration
    f1_score = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_curve, recall_curve)]

    # Append the precision and recall curves for this configuration
    precision_values_conf.append(precision_curve)
    recall_values_conf.append(recall_curve)
    f1_score_values_conf.append(f1_score)

    # Select the best 10 configurations based on F1 score
    best_config_indices = np.argsort(best_f1_scores)[-1:]

    # Plot precision and recall curves for the best configurations
    plt.figure(figsize=(10, 6))
    for i in best_config_indices:
        conf_gun, conf = best_configs[i]
        plt.plot(window_sizes, precision_values_conf[0], 
                 label=f'Precision (conf_gun=0.8, conf_shooter=0.6)', color='r')
        plt.plot(window_sizes, precision_values[i], 
                 label=f'Precision (conf_shooter={conf})', linestyle='dashed', color='r')
        plt.plot(window_sizes, recall_values_conf[0], 
                 label=f'Recall (conf_gun=0.8, conf_shooter=0.6)', color='b')
        plt.plot(window_sizes, recall_values[i], 
                 label=f'Recall (conf_shooter={conf})', linestyle='dashed', color='b')
        plt.plot(window_sizes, f1_score_values_conf[0], 
                label=f'F1 Score (conf_gun=0.8, conf_shooter=0.6)', color='g')
        plt.plot(window_sizes, f1_score_values[i], 
                label=f'F1 Score (conf_shooter={conf})', linestyle='dashed', color='g')

    plt.xlabel('Window Size')
    plt.ylabel('Value')
    plt.title('Precision, Recall, and F1 Score vs. Window Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/p_r_f1_vs_time_window_both.png')
    plt.show()