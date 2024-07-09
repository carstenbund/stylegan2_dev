sudo docker run --gpus all -it --shm-size 8G --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    cb/pytorch $1 $2 $3 $4 $5 $6 $7 $8
