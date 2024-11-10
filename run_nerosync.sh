if [ $# -gt 0 ]; then
    rootdir=$@
else
    rootdir=$(pwd)/data/NeRO/GlossySynthetic_blender/
fi
cases=$(find $rootdir -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
for case in ${cases[@]}; do
    echo "Running scene: $case"
    python train.py -s $rootdir/$case --eval -m output/nerosync/${case} -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512
    python render.py -m output/nerosync/${case} --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512
    # extract mesh
    gs-extract-mesh -m output/nerosync/${case} -o output/nerosync/${case}/test/ours_30000
    python metrics.py -m output/nerosync/${case}
done
