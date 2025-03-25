m = 4
n = 4

# batch_size = (m * (n - 1)) * 2
# batch_size = (2*m-1)*(2*n-1)
# batch_size = 1
batch_size = 16

'''
type 0: normal;             input shape: m * n
type 1: history_image;      input shape: m * n * total_move
type 2: history_sequence;   input shape: total_move * (m * n)
type 3: history_video;      input shape: total_move * m * n * 1
'''
args_CNN = {
    'num_of_generate_data_for_train': 1,
    'epochs': 5,
    'batch_size': batch_size,
    'verbose': True,
    'type': 1,
    'train': True,
    'load_model_name': None
}

args_Res = {
    'num_of_generate_data_for_train': 1024,
    'epochs': 50,
    'batch_size': batch_size,
    'verbose': True,
    'type': 0,
    'train': True,
    'load_model_name': None
}

args_LSTM = {
    'num_of_generate_data_for_train': 1024,
    'epochs': 50,
    'batch_size': batch_size,
    'verbose': True,
    'type': 2,
    'train': True,
    'load_model_name': None
}

args_ConvLSTM = {
    'num_of_generate_data_for_train': 100,
    'epochs': 16,
    'batch_size': batch_size,
    'verbose': True,
    'type': 3,
    'train': True,
    'load_model_name': None
}

args_Conv2Plus1D = {
    'num_of_generate_data_for_train': 100,
    'epochs': 10,
    'batch_size': batch_size,
    'verbose': True,
    'type': 3,
    'train': True,
    'load_model_name': None
}
