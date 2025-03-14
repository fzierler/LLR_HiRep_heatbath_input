#!/bin/bash
# Argument = -R NReplica
INPUTFILE="input_file_rep"
WDIR=$PWD
usage()
{
cat << EOF
usage: $0 options

This script run the LLR program on

OPTIONS:
   -h      Show this message
   -r      Number of replicas
EOF
}
r=
while getopts “hr:A:” OPTION
do
     case $OPTION in
         h)
             usage
             exit 1
             ;;
         r)
             r=$OPTARG
             ;;
         A)
             FILEA=$OPTARG
             ;;
         ?)
             usage
             exit
             ;;
     esac
done
if [[ -z $r ]]
then
     usage
     exit 1
fi

M=$(ls Rep_0/input_file* | wc -l)
i=0
while read -r line
do
    name="$line"
    stringarray=($line)
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

    cp Rep_${i}/input_file Rep_${i}/input_file_${M}
    cp Rep_${i}/rand_state Rep_${i}/rand_state_${M}
    sed -i "/rlx_seed /c\rlx_seed = ${RANDOM}" Rep_${i}/input_file
    sed -i "/rlx_start /c\rlx_start = rand_state" Rep_${i}/input_file
    sed -i "/gauge start = /c\gauge start = ${New_File}" Rep_${i}/input_file
    sed -i "/llr:S0 =/c\llr:S0 = $E" Rep_${i}/input_file
    sed -i "/llr:dS =/c\llr:dS = ${de}" Rep_${i}/input_file
    sed -i "/llr:starta =/c\llr:starta = ${A}" Rep_${i}/input_file
    sed -i "/llr:nfxa =/c\llr:nfxa = 0" Rep_${i}/input_file
    sed -i "/last conf =/c\last conf = +1" Rep_${i}/input_file
    sed -i "/llr:N_nr =/c\llr:N_nr = 0" Rep_${i}/input_file

    i=`echo "${i}+1"|bc -l`

done < "$FILEA"
