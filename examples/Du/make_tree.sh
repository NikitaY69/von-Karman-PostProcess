#!/bin/bash
mkdir logs
mkdir figs
mkdir figs/logs
mkdir misc/logs

dirs=(Du_l1 Du_2)
types=(bulk interior penal full)
for dir in ${dirs[@]}
do
    mkdir ${dir}
    mkdir figs/${dir}
    for type in ${types[@]}
    do
        mkdir ${dir}/${type}
    done
done