#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name changeSpacing.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="changeSpacing.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# From json file, read required variable.
readonly IMAGE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".image_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly LABEL_NAME=$(cat ${JSON_FILE} | jq -r ".label_name")
readonly SAVE_IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".save_image_name")
readonly SAVE_LABEL_NAME=$(cat ${JSON_FILE} | jq -r ".save_label_name")
readonly SPACING=$(cat ${JSON_FILE} | jq -r ".spacing")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")


for number in ${NUM_ARRAY[@]}
do
    image_path="${IMAGE_DIRECTORY}/case_${number}/${IMAGE_NAME}"
    label_path="${IMAGE_DIRECTORY}/case_${number}/${LABEL_NAME}"
    save_image_path="${SAVE_DIRECTORY}/case_${number}/${SAVE_IMAGE_NAME}"
    save_label_path="${SAVE_DIRECTORY}/case_${number}/${SAVE_LABEL_NAME}"

    echo "IMAGE_PATH:${image_path}"
    echo "LABEL_PATH:${label_path}"
    echo "SAVE_IMAGE_PATH:${save_image_path}"
    echo "SAVE_LABEL_PATH:${save_label_path}"
    echo "SPACING:${SPACING}"

    python3 changeSpacing.py ${image_path} ${save_image_path} --spacing ${SPACING}
    python3 changeSpacing.py ${label_path} ${save_label_path} --spacing ${SPACING} --is_label

# Judge if it works.
    if [ $? -eq 0 ]; then
     echo "Done."

    else
     echo "Fail"

    fi
done
