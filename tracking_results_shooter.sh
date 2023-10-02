#!/bin/bash

# Path to the folder containing model weights
model_weights_folder="weights"

# Range of values for conf_gun and overlap_gun
min_value=0.1
min_value_ov=0.5
max_value=0.9
step=0.1  # Adjust step size if needed

# Output folder for evaluation results
output_folder="evaluation_results_shooter"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Videos to evaluate
videos=("MOT-ASTERS-01.mp4" "MOT-ASTERS-02.mp4" "MOT-ASTERS-03.mp4" "MOT-ASTERS-04.mp4" "MOT-ASTERS-05.mp4" "MOT-ASTERS-06.mp4")
# videos=("MOT-ASTERS-02.mp4" "MOT-ASTERS-03.mp4")


# Iterate through model weights
for model_weight in $model_weights_folder/*; do
    model_name=$(basename "$model_weight")
    echo "Processing model weight: $model_name"
    
    # Iterate through conf values
    for conf in $(seq $min_value $step $max_value); do
        echo "Running experiments with conf=$conf in parallel"

        experiment_name="exp_${model_name}_conf${conf}"
        
        # Run evaluations for multiple videos in parallel
        for video in "${videos[@]}"; do
            # conf of 0.25 when varying overlap threshold, default overlap thresh is 0.75
            nohup python examples/track.py --save-mot --yolo-model "$model_weight" --source "inference/videos/${video}" --name "exp_${model_name}_video${video%.*}_conf${conf}" --device 0 --conf "$conf" --project Tracking_Outputs_shooter --tracking-method deepocsort --classes 0 &
        done
        
        # Wait for all parallel processes to finish
        wait

        # Move generated txt files to the test folder for the current experiment
        for experiment_folder in "Tracking_Outputs_shooter"/*; do
            
            # Create the destination directory
            mkdir -p "MOT-ASTERS/test_shooter/${experiment_name}/"
            
            # Iterate over the videos
            for video in "${videos[@]}"; do
                txt_file="${experiment_folder}/labels/${video%.*}.txt"
                
                # Check if the txt file exists
                if [ -e "$txt_file" ]; then
                    mv "$txt_file" "MOT-ASTERS/test_shooter/${experiment_name}/"
                else
                    touch "MOT-ASTERS/test_shooter/${experiment_name}/${video%.*}.txt"
                fi
            done
        done


        # Clear the Tracking_Outputs directory
        rm -rf Tracking_Outputs_shooter/*
        
        # Compute mot metrics and save console as a txt file
        mkdir -p "${output_folder}/${experiment_name}/"
        python -m motmetrics.apps.eval_motchallenge MOT-ASTERS/train/ "MOT-ASTERS/test_shooter/${experiment_name}" > "${output_folder}/${experiment_name}/metrics.txt"
    done
done