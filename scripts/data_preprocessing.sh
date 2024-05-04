#cd /media/allenyljiang/564AFA804AFA5BE5/Codes/hf2vad
cd /media/allenyljiang/564AFA804AFA5BE5/Codes/hf2vad/pre_process

#dataset=ped2 # shanghaitech
dataset_list=(shanghaitech ) # ped2:5:20开始,5:50结束
for dataset in "${dataset_list[@]}"
  do
  modelist=(train test)
    for part in "${modelist[@]}"

    do
      echo $part $dataset
      echo extract_bboxes
      python extract_bboxes.py --dataset_name $dataset --mode $part
      echo extract_flows
      python extract_flows.py --dataset_name $dataset --mode $part
      echo extract_samples
      python extract_samples.py --dataset_name $dataset --mode $part
    done
  done