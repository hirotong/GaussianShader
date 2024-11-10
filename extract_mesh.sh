exps=$(find ./output -maxdepth 1 -mindepth 1 -type d)
for exp in $exps; do
    echo "Extracting mesh from $exp"
    gs-extract-mesh -m "$exp" -o "$exp"
done