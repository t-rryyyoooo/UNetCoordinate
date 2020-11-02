#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name train.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="train.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

readonly IMAGE_PATH_LAYER_1=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_path_layer_1"))
readonly IMAGE_PATH_LAYER_2=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_path_layer_2"))
readonly IMAGE_PATH_THIN=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_path_thin"))
readonly LABEL_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".label_path"))
readonly MODEL_SAVEPATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".model_savepath"))
readonly TRAIN_LIST=$(cat ${JSON_FILE} | jq -r ".train_list")
readonly VAL_LIST=$(cat ${JSON_FILE} | jq -r ".val_list")
readonly LOG=$(eval echo $(cat ${JSON_FILE} | jq -r ".log"))
readonly IN_CHANNEL_1=$(cat ${JSON_FILE} | jq -r ".in_channel_1")
readonly IN_CHANNEL_2=$(cat ${JSON_FILE} | jq -r ".in_channel_2")
readonly IN_CHANNEL_THIN=$(cat ${JSON_FILE} | jq -r ".in_channel_thin")
readonly OUT_CHANNEL_THIN=$(cat ${JSON_FILE} | jq -r ".out_channel_thin")
readonly NUM_CLASS=$(cat ${JSON_FILE} | jq -r ".num_class")
readonly LEARNING_RATE=$(cat ${JSON_FILE} | jq -r ".learning_rate")
readonly BATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".batch_size")
readonly NUM_WORKERS=$(cat ${JSON_FILE} | jq -r ".num_workers")
readonly EPOCH=$(cat ${JSON_FILE} | jq -r ".epoch")
readonly GPU_IDS=$(cat ${JSON_FILE} | jq -r ".gpu_ids")
readonly API_KEY=$(cat ${JSON_FILE} | jq -r ".api_key")
readonly PROJECT_NAME=$(cat ${JSON_FILE} | jq -r ".project_name")
readonly EXPERIMENT_NAME=$(cat ${JSON_FILE} | jq -r ".experiment_name")

echo "IMAGE_PATH_LAYER_1:${IMAGE_PATH_LAYER_1}"
echo "IMAGE_PATH_LAYER_2:${IMAGE_PATH_LAYER_2}"
echo "IMAGE_PATH_THIN:${IMAGE_PATH_THIN}"
echo "LABEL_PATH:${LABEL_PATH}"
echo "MODEL_SAVEPATH:${MODEL_SAVEPATH}"
echo "TRAIN_LIST:${TRAIN_LIST}"
echo "VAL_LIST:${VAL_LIST}"
echo "LOG:${LOG}"
echo "IN_CHANNEL_1:${IN_CHANNEL_1}"
echo "IN_CHANNEL_2:${IN_CHANNEL_2}"
echo "IN_CHANNEL_THIN:${IN_CHANNEL_THIN}"
echo "OUT_CHANNEL_THIN:${OUT_CHANNEL_THIN}"
echo "NUM_CLASS:${NUM_CLASS}"
echo "LEARNING_RATE:${LEARNING_RATE}"
echo "BATCH_SIZE:${BATCH_SIZE}"
echo "NUM_WORKERS:${NUM_WORKERS}"
echo "EPOCH:${EPOCH}"
echo "GPU_IDS:${GPU_IDS}"
echo "API_KEY:${API_KEY}"
echo "PROJECT_NAME:${PROJECT_NAME}"
echo "EXPERIMENT_NAME:${EXPERIMENT_NAME}"

python3 train.py ${IMAGE_PATH_LAYER_1} ${IMAGE_PATH_LAYER_2} ${IMAGE_PATH_THIN} ${LABEL_PATH} ${MODEL_SAVEPATH} --train_list ${TRAIN_LIST} --val_list ${VAL_LIST} --log ${LOG} --in_channel_1 ${IN_CHANNEL_1} --in_channel_2 ${IN_CHANNEL_2} --in_channel_thin ${IN_CHANNEL_THIN} --out_channel_thin ${OUT_CHANNEL_THIN} --num_class ${NUM_CLASS} --lr ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --epoch ${EPOCH} --gpu_ids ${GPU_IDS} --api_key ${API_KEY} --project_name ${PROJECT_NAME} --experiment_name ${EXPERIMENT_NAME}
