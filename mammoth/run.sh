buffer=(500 2560 5120)
for sz in "${buffer[@]}"
do
    for i in {1..5}
    do
        echo $sz
        CUDA_VISIBLE_DEVICES=$1 python ./utils/main.py --model derpp --buffer_size $sz --dataset $3 -tinyimg --epochs 1 --nl $2 --csv_log
    done
done



