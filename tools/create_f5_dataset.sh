#!/bin/bash
if [ $# -ne 1 ]||[ ! -f $1 ];then
    echo "You must specify the source file."
    exit 1
fi
awk 'BEGIN { FS="|"; OFS="|" }{n = split($1, arr, "/");print(arr[n],$4);}' $1 > metadata.csv