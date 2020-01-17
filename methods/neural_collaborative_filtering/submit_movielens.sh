echo "Submitting jobs for "$1"..."
EPOCHS=15
echo "Submitting GMF jobs..."
bsub -n 1 -W 04:00 -R 'rusage[mem=4096]' -J $1"_gmf_32" -o "Results/"$1"/gmf_32.txt" python GMF.py --dataset $1 --epochs $EPOCHS --num_factors 32
echo "Submitting MLP jobs..."
bsub -n 1 -W 04:00 -R 'rusage[mem=4096]' -J $1"_mlp_64_32_16_8" -o "Results/"$1"/mlp_64_32_16_8.txt" python MLP.py --dataset $1 --epochs $EPOCHS --layers [64,32,16,8] --reg_layers [0,0,0,0]
echo "Submitting NeuMF jobs..."
bsub -n 1 -W 04:00 -R 'rusage[mem=4096]' -J $1"_neumf_32__16_8" -o "Results/"$1"/neumf_32__16_8.txt" python NeuMF.py --dataset $1 --epochs $EPOCHS --num_factors 32 --layers [16,8] --reg_layers [0,0]
echo "Done..."
