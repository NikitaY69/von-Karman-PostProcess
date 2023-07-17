#!/bin/bash
mkdir logs
mkdir figs
dirs=(Du_l1 Du_2)
types=(bulk interior penal full)
for dir in ${dirs[@]}
do
    mkdir ${dir}
    for type in ${types[@]}
    do
        mkdir ${dir}/${type}
        mkdir figs/${type}
    done
done