scenes=("car" "coffee" "helmet" "teapot" "toaster")
for scene in ${scenes[@]}; do
    echo "Running scene: $scene"
    CUDA_VISIBLE_DEVICES='1' python train.py -s data/refnerf/${scene} --eval -m output/${scene}  -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 --port 6001 &&
    CUDA_VISIBLE_DEVICES='1' python render.py -m output/${scene} --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512 &&
    python metrics.py -m output/${scene}
done