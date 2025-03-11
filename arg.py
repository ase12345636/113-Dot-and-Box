m = 4
n = 4

batch_size = (m * (n - 1)) * 2

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
    'type': 1
}

args_Res = {
    'num_of_generate_data_for_train': 1,
    'epochs': 5,
    'batch_size': batch_size,
    'verbose': True,
    'type': 0
}

args_LSTM = {
    'num_of_generate_data_for_train': 1,
    'epochs': 5,
    'batch_size': batch_size,
    'verbose': True,
    'type': 2
}
