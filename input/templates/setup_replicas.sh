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

i=0
while read -r line
do
    name="$line"
    stringarray=($line)
    A=${stringarray[1]}
    A=`echo "${A}"|bc -l`
    mkdir Rep_${i}
    mkdir Rep_${i}/Cnfg
    cp $INPUTFILE Rep_${i}/input_file
    RAN=`echo "${RANDOM}+10000"|bc -l`
    echo " " >> Rep_${i}/input_file
    echo "rlx_seed = $RAN" >> Rep_${i}/input_file
    de=${stringarray[2]}
    #E=`echo "${stringarray[0]} + ${de}*0.5" | bc -l`
    E=${stringarray[0]}
    echo "llr:S0 = $E" >> Rep_${i}/input_file
    echo "llr:dS = ${de}" >> Rep_${i}/input_file
    echo "llr:starta = ${A}" >> Rep_${i}/input_file
    sed -i "/llr:Smin =/d" Rep_${i}/input_file
    echo "llr:Smin = ${Emin}" >> Rep_${i}/input_file
    sed -i "/llr:Smax =/d" Rep_${i}/input_file
    echo "llr:Smax = ${Emax}" >> Rep_${i}/input_file
    i=`echo "${i}+1"|bc -l`

done < "$FILEA"
mkdir CSV
