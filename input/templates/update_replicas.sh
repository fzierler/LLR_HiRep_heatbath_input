#!/bin/bash
r=
while getopts “r:i:” OPTION
do
     case $OPTION in
         r)
             r=$OPTARG
             ;;
         i)
             FILEA=$OPTARG
             ;;
     esac
done
if [[ -z $r ]]; then
     echo "Number of replicas -r not specified"
     exit 1
fi
if [[ -z $FILEA ]]; then
     echo "Input file -i not specified"
     exit 1
fi

for (( i=0; i<$r; i+=1 )); do

    # find all configurations in relevant directory
    confs=$(find ./Rep_${i} -name run1*)
    if [[ -n $confs ]]; then
        gsfile=$(ls Rep_${i}/run1* -t | head -1)
    fi
    
    if [[ -f Rep_$i/out_0 ]]; then 
        RM_NUM=$(grep 'Robbins Monro sequence #' Rep_0/out_0 | tail -n 1 | grep -oP '(?<=#).*?(?=:)')
        de=$(grep "LLR Delta S" Rep_$i/out_0 | grep -o -E '[0-9]+(\.[0-9]+)'| tail -n 1)
        E=$(grep "a_rho(0," Rep_$i/out_0 | tail -1 | grep -o -E '[0-9]+(\.[0-9]+)'| head -n 1)
        A=$(grep "a_rho(0," Rep_$i/out_0 | tail -1 | grep -o -E '[0-9]+(\.[0-9]+)'| tail -n 1)
    fi

    sed -i "/rlx_seed/c\rlx_seed = ${RANDOM}"         Rep_${i}/$FILEA
    # For the following quantities first test if we were able to read the required information from the logs
    if [[ -n ${gsfile} ]]; then
        gsfile=${gsfile#"Rep_${i}/"}
        sed -i "/gauge start/c\gauge start = ${gsfile}" Rep_${i}/$FILEA
    fi
    if [[ -n ${de} ]]; then
        sed -i "/llr:dS/c\llr:dS = ${de}"             Rep_${i}/$FILEA
    fi
    if [[ -n $E ]]; then
        sed -i "/llr:S0/c\llr:S0 = $E"                Rep_${i}/$FILEA
    fi
    if [[ -n $A ]]; then
        sed -i "/llr:starta/c\llr:starta = ${A}"      Rep_${i}/$FILEA
    fi
    if [[ -n $RM_NUM ]]; then
        # increment by one to get the next Robbins-Monro weight.
        sed -i "/llr:it =/c\llr:it = $(($RM_NUM+1))"  Rep_${i}/$FILEA
    fi

    # remove old configuration files
    if [[ -n $confs ]]; then
        # check if all files are of the same size
        same_size=$(readlink -f Rep_${i}/run1* | xargs du | awk '{print $1}' | uniq -u | wc -l)
        # same_size is equal to zero if all files are of the same size
        # only if this is the case this script will remove old configurations
        if [ $same_size -eq 0 ]; then
            readlink -f Rep_${i}/run1* >> tmp_list
            if [ $(wc -l tmp_list | awk '{print $1}') -gt 1 ]; then
                cat tmp_list | sort -V | head -n -1 | xargs rm
            fi
        rm tmp_list
        fi
    fi

    # if no output files exists, create one
    # (this fixes issues seen on sunbird, where HiRep gets stuck creating the files)
    touch Rep_$i/out_0 

done
