#!/bin/bash
# Argument = -R NReplica
INPUTFILE="input_file_rep_start"
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

i=0
while read -r line
do
    stringarray=($line)
    E=${stringarray[0]}
    A=${stringarray[1]}
    de=${stringarray[2]}
    mkdir Rep_${i}
    cp  $INPUTFILE                            Rep_${i}/input_file_start
    sed -i "/rlx_seed/c\rlx_seed = ${RANDOM}" Rep_${i}/input_file_start
    sed -i "/llr:S0/c\llr:S0 = $E"            Rep_${i}/input_file_start
    sed -i "/llr:dS/c\llr:dS = ${de}"         Rep_${i}/input_file_start
    sed -i "/llr:starta/c\llr:starta = ${A}"  Rep_${i}/input_file_start
    sed -i "/llr:Smin/c\llr:Smin = ${Emin}"   Rep_${i}/input_file_start
    sed -i "/llr:Smax/c\llr:Smax = ${Emax}"   Rep_${i}/input_file_start

    i=`echo "${i}+1"|bc -l`

done < "$FILEA"
