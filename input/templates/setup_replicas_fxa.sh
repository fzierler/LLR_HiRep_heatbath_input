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
    gsfile=$(ls Rep_${i}/run1* -t | head -1)
    gsfile=${gsfile#"Rep_${i}/"}
    RM_NUM=$(grep 'Robbins Monro sequence #' Rep_0/out_0 | tail -n 1 | grep -oP '(?<=#).*?(?=:)')
    if ! [[ -n $RM_NUM ]]
    then
        RM_NUM=0
    fi
    New_File=$(echo $gsfile | grep -oP '.*(?<=n)')
    New_File=$(echo $New_File$RM_NUM)
    if [ "$gsfile" != "$New_File" ]; then
        mv Rep_${i}/$gsfile Rep_${i}/$New_File
    fi
    
    de=$(grep "LLR Delta S" Rep_$i/out_0 | grep -o -E '[0-9]+(\.[0-9]+)'| tail -n 1)
    E=$(grep "a_rho(0," Rep_$i/out_0 | tail -1 | grep -o -E '[0-9]+(\.[0-9]+)'| head -n 1)
    A=$(grep "a_rho(0," Rep_$i/out_0 | tail -1 | grep -o -E '[0-9]+(\.[0-9]+)'| tail -n 1) 

    cp Rep_${i}/rand_state       Rep_${i}/rand_state_3
    cp Rep_${i}/input_file_cont  Rep_${i}/input_file_3
    cp input_file_fxa_rep        Rep_${i}/input_file_fxa

    sed -i "/rlx_seed/c\rlx_seed = ${RANDOM}"         Rep_${i}/input_file_fxa
    sed -i "/gauge start/c\gauge start = ${New_File}" Rep_${i}/input_file_fxa
    sed -i "/llr:S0/c\llr:S0 = $E"                    Rep_${i}/input_file_fxa
    sed -i "/llr:dS/c\llr:dS = ${de}"                 Rep_${i}/input_file_fxa
    sed -i "/llr:starta/c\llr:starta = ${A}"          Rep_${i}/input_file_fxa
    # Everything below does never change
    sed -i "/rlx_start/c\rlx_start = rand_state"  Rep_${i}/input_file_fxa
    sed -i "/llr:nfxa/c\llr:nfxa = 50"            Rep_${i}/input_file_fxa
    sed -i "/last conf/c\last conf = +0"          Rep_${i}/input_file_fxa
    sed -i "/llr:N_nr/c\llr:N_nr = 0"             Rep_${i}/input_file_fxa
    sed -i "/llr:sfreq_fxa/c\llr:sfreq_fxa = 100" Rep_${i}/input_file_fxa

done
