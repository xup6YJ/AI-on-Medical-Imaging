


#alpha
# python inference.py \
# --model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_elu_0.2_84.54.pt'

# python inference.py \
# --model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_elu_0.6_82.22.pt'

# python inference.py \
# --model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_elu_1.0_83.89.pt'

# python inference.py \
# --model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_elu_1.4_81.67.pt'

# python inference.py \
# --model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_elu_1.8_74.26.pt'

#dropout
python inference.py \
--model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_elu_1.0_83.89.pt'

python inference.py \
--model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.5_elu_1.0_79.44.pt'

python inference.py \
--model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.75_elu_1.0_74.35.pt'


#activation
python inference.py \
--model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_elu_1.0_83.89.pt'

python inference.py \
--model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_relu_1.0_85.37.pt'

python inference.py \
--model_path '/home/yclin/Documents/MedAI/lab2/weights/EEGNet_drop0.25_leakyrelu_1.0_84.35.pt'