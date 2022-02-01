#
LR=0.001
NUMBER_SAMPLES=1




LATENT_LIST="256"
BETA_LIST="1e-10 1e-09 1e-08 1e-07 1e-06 1e-05 0.0001 0.001 0.01 0.1"



for LATENT in $LATENT_LIST ; do
for BETA in $BETA_LIST ; do


python pretrain.py --latent $LATENT --beta_value $BETA --lr $LR
python imprint.py --model $LATENT/$BETA/pretrain_checkpoint/model_best.pth.tar --beta_value $BETA --latent $LATENT --num-sample $NUMBER_SAMPLES

done;
done;


