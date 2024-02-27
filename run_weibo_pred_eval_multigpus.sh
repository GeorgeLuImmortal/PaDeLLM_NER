DATASET="weibo"
IDENTIFIER="bs128_e4"
MODEL_SIZE="7b"
CUDA_NUM="1"
SAMPLE_METHOD="greedy"


echo "===="
echo "starting theory inference $DATASET"
echo "===="

for RUN in 1
do
    python3 infer/padellm_multigpus.py \
    --dataset $DATASET \
    --model_size $MODEL_SIZE \
    --identifier $IDENTIFIER \
    --run $RUN \
    --cuda_num $CUDA_NUM
done

VARIANT="${IDENTIFIER}_"${SAMPLE_METHOD}"_theory_run1"
python3 eval/dedu.py --dataset $DATASET --model_size $MODEL_SIZE --identifier $VARIANT
python3 eval/eval.py --dataset $DATASET --model_size $MODEL_SIZE --identifier $VARIANT
python3 eval/eval_dedu.py --dataset $DATASET --model_size $MODEL_SIZE --identifier $VARIANT

