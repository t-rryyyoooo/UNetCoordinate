#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name segmentation.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="segmentation.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly WEIGHT=$(eval echo $(cat ${JSON_FILE} | jq -r ".weight"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".image_patch_size")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly OVERLAP=$(cat ${JSON_FILE} | jq -r ".overlap")
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly SAVE_NAME=$(cat ${JSON_FILE} | jq -r ".save_name")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")
readonly GPU_ID=$(cat ${JSON_FILE} | jq -r ".gpu_id")

echo $NUM_ARRAY
for number in ${NUM_ARRAY[@]}
do
 save="${SAVE_DIRECTORY}/case_${number}/${SAVE_NAME}"
 image="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
 mask="${DATA_DIRECTORY}/case_${number}/${MASK_NAME}"

 echo "Image:${image}"
 echo "WEIGHT:${WEIGHT}"
 echo "Mask:${mask}"
 echo "Save:${save}"
 echo "IMAGE_PATCH_SIZE:${IMAGE_PATCH_SIZE}"
 echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"
 echo "OVERLAP:${OVERLAP}"
 echo "GPU_ID:${GPU_ID}"

 python3 segmentation.py $image $WEIGHT $save --mask_path $mask --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --overlap $OVERLAP -g ${GPU_ID}


done
