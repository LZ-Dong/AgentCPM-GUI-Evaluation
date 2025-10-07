# export CKPT="GUI-Owl-7B"   # 模型目录或 HF 模型名
# export PIXEL_ARGS='{"min_pixels":3136,"max_pixels":10035200}'
# export IMAGE_LIMIT_ARGS='image=2'
# export MP_SIZE=1
# MM_KWARGS=(
#     --mm-processor-kwargs "$PIXEL_ARGS"
#     --limit-mm-per-prompt "$IMAGE_LIMIT_ARGS"
# )

# CUDA_VISIBLE_DEVICES=6 vllm serve $CKPT \
#     --max-model-len 32768 "${MM_KWARGS[@]}" \
#     --tensor-parallel-size $MP_SIZE \
#     --allowed-local-media-path '/' \
#     --port 4243


export CKPT="GUI-Owl-32B"   # 模型目录或 HF 模型名
export PIXEL_ARGS='{"min_pixels":3136,"max_pixels":10035200}'
export IMAGE_LIMIT_ARGS='image=2'
export MP_SIZE=4   # 4张卡并行

MM_KWARGS=(
    --mm-processor-kwargs "$PIXEL_ARGS"
    --limit-mm-per-prompt "$IMAGE_LIMIT_ARGS"
)

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve $CKPT \
    --max-model-len 32768 "${MM_KWARGS[@]}" \
    --tensor-parallel-size $MP_SIZE \
    --allowed-local-media-path '/' \
    --port 4243
