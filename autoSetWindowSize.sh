#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name setWindowSize.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="setWindowSize.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly SAVE_NAME=$(cat ${JSON_FILE} | jq -r ".save_name")
readonly MIN_VALUE=$(cat ${JSON_FILE} | jq -r ".min_value")
readonly MAX_VALUE=$(cat ${JSON_FILE} | jq -r ".max_value")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")

echo $NUM_ARRAY
for number in ${NUM_ARRAY[@]}
do
 image_path="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
 save_path="${SAVE_DIRECTORY}/case_${number}/${SAVE_NAME}"

 echo "image_path:${image_path}"
 echo "save_path:${save_path}"
 echo "min_value:${MIN_VALUE}"
 echo "max_value:${MAX_VALUE}"

 python3 setWindowSize.py ${image_path} ${save_path} --min_value ${MIN_VALUE} --max_value ${MAX_VALUE}

done
