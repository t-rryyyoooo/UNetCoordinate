#!/bin/bash

#Input
# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name makeMaskImage.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="makeMaskImage.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

# From json file, read required variables.
readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly LABEL_NAME=$(cat ${JSON_FILE} | jq -r ".label_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly MASK_NUM=$(cat ${JSON_FILE} | jq -r ".mask_num")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")
readonly LOG_FILE=$(cat ${JSON_FILE} | jq -r ".log_file")

mkdir -p `dirname ${LOG_FILE}`
date >> $LOG_FILE

for number in ${NUM_ARRAY[@]}
do

 data="${DATA_DIRECTORY}/case_${number}"
 label="${data}/${LABEL_NAME}"
 save="${data}/${MASK_NAME}"

 echo "LABEL:$label"
 echo "SAVE:$save"
 echo "MASK_NUM:$MASK_NUM"

 python3 makeMaskImage.py ${label} ${save} --mask_number ${MASK_NUM}

 # Judge if it works.
 if [ $? -eq 0 ]; then
  echo "case_${number} done."

 else
  echo "case_${number}" >> $LOG_FILE
  echo "case_${number} failed"

 fi

done


