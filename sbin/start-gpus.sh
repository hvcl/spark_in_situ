#!/bin/bash
#mpirun --mca oob_tcp_if_include ib0 -np 4 -host ib1,ib2,ib3,ib4 python gpu_manager.py

#PYGPU_MANAGER="python $SPARK_HOME/python/pyspark/vislib/gpu_manager.py"
PYGPU_MANAGER=/home/smhong/spark_hvcl/bin/gpu_manager
SLAVES_FILE=/home/smhong/spark_hvcl/conf/slaves
#PREFIX="--prefix $LOCAL_HOME"
MPIRUN=`which mpirun`
pid=/tmp/vispark_worker.pid

#echo $MPIRUN 

NP=0
HOST=""
declare -a HOST


while read line
do 
    if [[ $line != *"#"* ]]; then 
        if [[ ${#line} > 1 ]]; then 
            #echo $line
            NP=$(( NP+1 ))
            HOST+=("$line")
            #HOST=$HOST$line, 
        fi
    fi
done < $SLAVES_FILE

echo $NP
echo $HOST

HOSTLIST=""

for x in ${HOST[@]}
do
    HOSTLIST+="$x,"
done

echo $HOSTLIST

MPI_OPTION=""

mpirun $MPI_OPTION -np $NP -host $HOSTLIST $PYGPU_MANAGER $SLAVES_FILE $pid
