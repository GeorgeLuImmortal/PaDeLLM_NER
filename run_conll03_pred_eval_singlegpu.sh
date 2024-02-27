DATASET="conll03"
IDENTIFIER="bs128_e4"
MODEL_SIZE="7b"
SAMPLE_METHOD="greedy"
CUDA_NUM="0"


echo "===="
echo "starting single inference $DATASET"
echo "===="

for RUN in 1
do
    python3 infer/padellm_singlegpu_eng.py \
    --dataset $DATASET \
    --model_size $MODEL_SIZE \
    --identifier $IDENTIFIER \
    --run $RUN \
    --cuda_num $CUDA_NUM \
    --sample_method $SAMPLE_METHOD
done

VARIANT="${IDENTIFIER}_"${SAMPLE_METHOD}"_single_run1"
python3 eval/dedu.py --dataset $DATASET --model_size $MODEL_SIZE --identifier $VARIANT
python3 eval/eval.py --dataset $DATASET --model_size $MODEL_SIZE --identifier $VARIANT
python3 eval/eval_dedu.py --dataset $DATASET --model_size $MODEL_SIZE --identifier $VARIANT
