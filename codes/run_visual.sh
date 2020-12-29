# set degradation_type (impulse or bicubic)
export CUDA_VISIBLE_DEVICES=0
MODEL="EDVR"
DEG_TYPE="bicubic"
EXP_NAME="test"
YML_NAME="EDVR_R.yml"

# Gaussian8 evaluation

# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type preset

# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 0.8 --sigma_y 0.8 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 0.9 --sigma_y 0.9 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 1.0 --sigma_y 1.0 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 1.1 --sigma_y 1.1 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 1.2 --sigma_y 1.2 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 1.3 --sigma_y 1.3 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 1.4 --sigma_y 1.4 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 1.5 --sigma_y 1.5 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 1.6 --sigma_y 1.6 --theta 0

# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 2.0 --sigma_y 2.0 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 3.0 --sigma_y 3.0 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type ${DEG_TYPE} --sigma_x 4.0 --sigma_y 4.0 --theta 0

# Anisotropic Gaussian evaluation
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type impulse --sigma_x 0.8 --sigma_y 1.6 --theta 0
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type impulse --sigma_x 0.8 --sigma_y 1.6 --theta 45
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type impulse --sigma_x 0.8 --sigma_y 1.6 --theta 90
# python test_dynavsr.py -opt options/test/${MODEL}/${YML_NAME} --exp_name ${EXP_NAME} --degradation_type impulse --sigma_x 0.8 --sigma_y 1.6 --theta 135

# Demo
# python test_dynavsr.py -opt options/test/EDVR/EDVR_Demo.yml --exp_name Demo