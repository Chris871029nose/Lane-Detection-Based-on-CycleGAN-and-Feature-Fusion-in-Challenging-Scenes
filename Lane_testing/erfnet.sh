python3 -u test_erfnet.py CULane ERFNet train_gt test_img \
                          --lr 0.01 \
                          --gpus 0 1 \
                          --resume trained/model_A.pth.tar \
                          --img_height 208 \
                          --img_width 976 \
                          -j 16 \
                          -b 40
