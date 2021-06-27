for BEHAVIOUR_SIGMA in "0.5" "1.0" "2.0"
do
    for BASELINERATE in "5" "10" "15"
    do
        python latentneural -r "latentneural/data/storage/lorenz/grid/train_trials=50|baselinerate=$BASELINERATE|behaviour_sigma=$BEHAVIOUR_SIGMA/training_settings.yaml"
    done
done