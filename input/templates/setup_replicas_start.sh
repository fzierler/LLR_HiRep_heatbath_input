#!/bin/bash
# Argument = -R NReplica
r=
while getopts “r:” OPTION
do
     case $OPTION in
         r)
             r=$OPTARG
             ;;
     esac
done
if [[ -z $r ]]
then
     echo "Number of replicas -r not specified"
     exit 1
fi

for (( i=0; i<$r; i+=1 )); do

    sed -i "/rlx_seed/c\rlx_seed = ${RANDOM}" Rep_${i}/input_file_start

done