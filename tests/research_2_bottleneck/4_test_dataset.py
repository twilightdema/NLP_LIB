from task_util import get_dataset_loader_func

load_encoded_cola_data_spm, load_encoded_cola_data_static_word_embedding = get_dataset_loader_func("cola")
load_encoded_mrpc_data_spm, load_encoded_mrpc_data_static_word_embedding = get_dataset_loader_func("mrpc")

data_cola_spm_train, data_cola_spm_dev = load_encoded_cola_data_spm(True)
data_cola_embed_train, data_cola_embed_dev = load_encoded_cola_data_static_word_embedding(True)
data_mrpc_spm_train, data_mrpc_spm_dev = load_encoded_mrpc_data_spm(True)
data_mprc_embed_train, data_mrpc_embed_dev = load_encoded_mrpc_data_static_word_embedding(True)

print('OK')
