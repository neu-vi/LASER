mkdir weights
cd weights

# SALAD (~ 350 MiB)
echo "Downloading SALAD weights..."
SALAD_URL="https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
curl -L "$SALAD_URL" -o "./dino_salad.ckpt"

# DINO (~ 340 MiB)
echo "Downloading DINO weights..."
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

cd ..