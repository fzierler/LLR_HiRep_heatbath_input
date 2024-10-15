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

read -r -a stringarray <<< $(head -n 1 $FILEA)
Emin=${stringarray[0]}
read -r -a stringarray <<< $(tail -n 1 $FILEA)
Emax=${stringarray[0]}

M=$(ls Rep_0/input_file* | wc -l)
i=0
while read -r line
do
    name="$line"
    stringarray=($line)
    A=${stringarray[1]}
    A=`echo "${A}"|bc -l`
    RAN=`echo "${RANDOM}+10000"|bc -l`
    cp Rep_${i}/input_file Rep_${i}/input_file_${M}
    echo " " >> Rep_${i}/input_file
    sed -i "/rlx_seed /d" Rep_${i}/input_file
    echo "rlx_seed = $RAN" >> Rep_${i}/input_file
    sed -i "/rlx_start /d" Rep_${i}/input_file
    echo "rlx_start = rand_state" >> Rep_${i}/input_file
    sed -i "/gauge start = /d" Rep_${i}/input_file
    gsfile=$(ls Rep_${i}/run1* -t | head -1)
    gsfile=${gsfile#"Rep_${i}/"}
    RM_NUM=$(grep 'Robbins Monro sequence #' Rep_0/out_0 | tail -n 1 | grep -oP '(?<=#).*?(?=:)')
    if ! [[ -n $RM_NUM ]]
    then
        RM_NUM=0
    fi
    New_File=$(echo $gsfile | grep -oP '.*(?<=n)')
    New_File=$(echo $New_File$RM_NUM)
    mv Rep_${i}/$gsfile Rep_${i}/$New_File
    echo "gauge start = ${New_File}" >> Rep_${i}/input_file
    cp Rep_${i}/rand_state Rep_${i}/rand_state_${M}
    N=$(grep "last conf =" Rep_$i/input_file | grep -o -E '[0-9]+'| tail -n 1)
    de=$(grep "LLR Delta S" Rep_$i/out_0 | grep -o -E '[0-9]+(\.[0-9]+)'| tail -n 1)
    E=$(grep "a_rho(0," Rep_$i/out_0 | tail -1 | grep -o -E '[0-9]+(\.[0-9]+)'| head -n 1)
    A=$(grep "a_rho(0," Rep_$i/out_0 | tail -1 | grep -o -E '[0-9]+(\.[0-9]+)'| tail -n 1) 

    sed -i "/llr:S0 =/d" Rep_${i}/input_file
    echo "llr:S0 = $E" >> Rep_${i}/input_file
    sed -i "/llr:dS =/d" Rep_${i}/input_file
    echo "llr:dS = ${de}" >> Rep_${i}/input_file
    sed -i "/llr:starta =/d" Rep_${i}/input_file
    echo "llr:starta = ${A}" >> Rep_${i}/input_file
    sed -i "/llr:nfxa =/d" Rep_${i}/input_file
    echo "llr:nfxa = 0" >> Rep_${i}/input_file
    sed -i "/last conf = +/d" Rep_${i}/input_file
    echo "last conf = 0" >> Rep_${i}/input_file
    sed -i "/llr:N_nr =/d" Rep_${i}/input_file
    echo "llr:N_nr = 1" >> Rep_${i}/input_file


    i=`echo "${i}+1"|bc -l`

done < "$FILEA"
