python3 -u train_erfnet.py CULane ERFNet train_gt2 val_gt \
                        --lr 0.01 \
                        --gpus 0 1 \
                        -j 16 \
                        -b 28 \
                        --epochs 42 \
                        --img_height 224 \
                        --img_width 992 \
2>&1|tee train_erfnet_culane.log

                        #--resume pretrained/ERFNet_pretrained.tar \
