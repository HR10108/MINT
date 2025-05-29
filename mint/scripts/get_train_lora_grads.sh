# #!/bin/bash
train_file=$1 
model=$2 # path to model
output_path=$3 # path to output
dims=$4 # dimension of projection, can be a list
gradient_type=$5
INFO_TYPE=$6
TASK_NAME=$7


if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m mint.get_grads.get_info \
--train_file $train_file \
--info_type $INFO_TYPE \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type $gradient_type \
--task $TASK_NAME
