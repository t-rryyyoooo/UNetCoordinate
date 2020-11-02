#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name caluculateDICE.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="caluculateDICE.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# From json file, read required variable.
readonly TRUE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".true_directory"))
readonly PREDICT_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".predict_directory"))
readonly SAVE_PATH=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_path"))
readonly PATIENTID_LIST=$(cat ${JSON_FILE} | jq -r ".patientID_list")
readonly CLASSES=$(cat ${JSON_FILE} | jq -r ".classes")
readonly CLASS_LABEL=$(cat ${JSON_FILE} | jq -r ".class_label")
readonly TRUE_NAME=$(cat ${JSON_FILE} | jq -r ".true_name")
readonly PREDICT_NAME=$(cat ${JSON_FILE} | jq -r ".predict_name")

echo "TRUE_DIRECTORY:${TRUE_DIRECTORY}"
echo "PREDICT_DIRECTORY:${PREDICT_DIRECTORY}"
echo "SAVE_PATH:${SAVE_PATH}"
echo "PATIENTID_LIST:${PATIENTID_LIST}"
echo "CLASSES:${CLASSES}"
echo "CLASS_LABEL:${CLASS_LABEL}"
echo "TRUE_NAME:${TRUE_NAME}"
echo "PREDICT_NAME:${PREDICT_NAME}"


python3 caluculateDICE.py ${TRUE_DIRECTORY} ${PREDICT_DIRECTORY} ${SAVE_PATH} ${PATIENTID_LIST} --classes ${CLASSES} --class_label ${CLASS_LABEL} --true_name ${TRUE_NAME} --predict_name ${PREDICT_NAME} 

# Judge if it works.
if [ $? -eq 0 ]; then
 echo "Done."

else
 echo "Fail"

fi

