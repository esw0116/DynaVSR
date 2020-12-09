# set degradation_mode (impulse or bicubic)
export CUDA_VISIBLE_DEVICES=1
DATA="REDS" # Vid4
MY_MODEL='MFDN'

#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode preset

#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 0.8
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 0.9
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 1.0
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 1.1
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 1.2
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 1.3
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 1.4
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 1.5
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 1.6

#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 0.8
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 0.9
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 1.0
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 1.1
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 1.2
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 1.3
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 1.4
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 1.5
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode bicubic --sigma_x 1.6

#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 0.0
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 45.0
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 90.0
#python make_slr_images.py -m ${DATA} --model ${MY_MODEL} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 135.0

#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 0.8
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 0.9
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 1.0
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 1.1
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 1.2
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 1.3
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 1.4
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 1.5
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 1.6

#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 0.8
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 0.9
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 1.0
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 1.1
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 1.2
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 1.3
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 1.4
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 1.5
#python make_downgraded_images.py -m ${DATA} --degradation_mode bicubic --sigma_x 1.6

#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 0.0
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 45.0
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 90.0
#python make_downgraded_images.py -m ${DATA} --degradation_mode impulse --sigma_x 0.8 --sigma_y 1.6 --theta 135.0
