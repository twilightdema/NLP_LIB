import json

class ConfigMapper:
  def get_config_path_for(obj, config_str):

    tokens = config_str.split('+')
    if len(tokens) == 1:
      # Transformer Decoder LM
      if config_str == 'tf-dec' or config_str == 'tf6-dec':
        return 'train_best2010_lm_tfbase_dec_s2s.json'
      elif config_str == 'tf2-dec':
        return 'train_best2010_lm_tfbase2_dec_s2s.json'
      elif config_str == 'tf4-dec':
        return 'train_best2010_lm_tfbase4_dec_s2s.json'
      elif config_str == 'tf12-dec':
        return 'train_best2010_lm_tfbase12_dec_s2s_lrscale0.25.json'
      elif config_str == 'tf-dec-bigram' or config_str == 'tf6-dec-bigram':
        return 'train_best2010_lm_tfbase_dec_s2s_bigram.json'
      elif config_str == 'tf2-dec-bigram':
        return 'train_best2010_lm_tfbase2_dec_s2s_bigram.json'
      elif config_str == 'tf4-dec-bigram':
        return 'train_best2010_lm_tfbase4_dec_s2s_bigram.json'
      elif config_str == 'tf12-dec-bigram':
        return 'train_best2010_lm_tfbase12_dec_s2s_bigram_lrscale0.25.json'
      elif config_str == 'tf-dec-sp' or config_str == 'tf6-dec-sp':
        return 'train_best2010_lm_tfbase_dec_s2s_sp.json'
      elif config_str == 'tf2-dec-sp':
        return 'train_best2010_lm_tfbase2_dec_s2s_sp.json'
      elif config_str == 'tf4-dec-sp':
        return 'train_best2010_lm_tfbase4_dec_s2s_sp.json'
      elif config_str == 'tf12-dec-sp':
        return 'train_best2010_lm_tfbase12_dec_s2s_sp_lrscale0.25.json'

      # Transformer Encoder LM
      elif config_str == 'tf-enc' or config_str == 'tf6-enc':
        return 'train_best2010_lm_tfbase_enc_s2s.json'
      elif config_str == 'tf2-enc':
        return 'train_best2010_lm_tfbase2_enc_s2s.json'
      elif config_str == 'tf4-enc':
        return 'train_best2010_lm_tfbase4_enc_s2s.json'
      elif config_str == 'tf12-enc':
        return 'train_best2010_lm_tfbase12_enc_s2s_lrscale0.25.json'
      elif config_str == 'tf-enc-bigram' or config_str == 'tf6-enc-bigram':
        return 'train_best2010_lm_tfbase_enc_s2s_bigram_lrscale0.25.json'
      elif config_str == 'tf2-enc-bigram':
        return 'train_best2010_lm_tfbase2_enc_s2s_bigram.json'
      elif config_str == 'tf4-enc-bigram':
        return 'train_best2010_lm_tfbase4_enc_s2s_bigram.json'
      elif config_str == 'tf12-enc-bigram':
        return 'train_best2010_lm_tfbase12_enc_s2s_bigram_lrscale0.10.json'
      elif config_str == 'tf-enc-sp' or config_str == 'tf6-enc-sp':
        return 'train_best2010_lm_tfbase_enc_s2s_sp.json'
      elif config_str == 'tf2-enc-sp':
        return 'train_best2010_lm_tfbase2_enc_s2s_sp.json'
      elif config_str == 'tf4-enc-sp':
        return 'train_best2010_lm_tfbase4_enc_s2s_sp.json'
      elif config_str == 'tf12-enc-sp':
        return 'train_best2010_lm_tfbase12_enc_s2s_sp_lrscale0.10.json'

    elif len(tokens) == 2:
      config_str = tokens[0]
      finetune_dataset = tokens[1]

      # Transformer Decoder Finetune
      if config_str == 'tf-dec' or config_str == 'tf6-dec':
        return 'finetune_best2010lm_tfbase_dec_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf2-dec':
        return 'finetune_best2010lm_tfbase2_dec_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf4-dec':
        return 'finetune_best2010lm_tfbase4_dec_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf12-dec':
        return 'finetune_best2010lm_tfbase12_dec_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf-dec-bigram' or config_str == 'tf6-dec-bigram':
        return 'finetune_best2010lm_tfbase_dec_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf2-dec-bigram':
        return 'finetune_best2010lm_tfbase2_dec_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf4-dec-bigram':
        return 'finetune_best2010lm_tfbase4_dec_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf12-dec-bigram':
        return 'finetune_best2010lm_tfbase12_dec_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf-dec-sp' or config_str == 'tf6-dec-sp':
        return 'finetune_best2010lm_tfbase_dec_s2s_sp_' + finetune_dataset + '.json'
      elif config_str == 'tf2-dec-sp':
        return 'finetune_best2010lm_tfbase2_dec_s2s_sp_' + finetune_dataset + '.json'
      elif config_str == 'tf4-dec-sp':
        return 'finetune_best2010lm_tfbase4_dec_s2s_sp_' + finetune_dataset + '.json'
      elif config_str == 'tf12-dec-sp':
        return 'finetune_best2010lm_tfbase12_dec_s2s_sp_' + finetune_dataset + '.json'

      # Transformer Encoder Finetune
      if config_str == 'tf-enc' or config_str == 'tf6-enc':
        return 'finetune_best2010lm_tfbase_enc_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf2-enc':
        return 'finetune_best2010lm_tfbase2_enc_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf4-enc':
        return 'finetune_best2010lm_tfbase4_enc_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf12-enc':
        return 'finetune_best2010lm_tfbase12_enc_s2s_' + finetune_dataset + '.json'
      elif config_str == 'tf-enc-bigram' or config_str == 'tf6-enc-bigram':
        return 'finetune_best2010lm_tfbase_enc_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf2-enc-bigram':
        return 'finetune_best2010lm_tfbase2_enc_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf4-enc-bigram':
        return 'finetune_best2010lm_tfbase4_enc_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf12-enc-bigram':
        return 'finetune_best2010lm_tfbase12_enc_s2s_bigram_' + finetune_dataset + '.json'
      elif config_str == 'tf-enc-sp' or config_str == 'tf6-enc-sp':
        return 'finetune_best2010lm_tfbase_enc_s2s_sp_' + finetune_dataset + '.json'
      elif config_str == 'tf2-enc-sp':
        return 'finetune_best2010lm_tfbase2_enc_s2s_sp_' + finetune_dataset + '.json'
      elif config_str == 'tf4-enc-sp':
        return 'finetune_best2010lm_tfbase4_enc_s2s_sp_' + finetune_dataset + '.json'
      elif config_str == 'tf12-enc-sp':
        return 'finetune_best2010lm_tfbase12_enc_s2s_sp_' + finetune_dataset + '.json'

    return None

  def construct_json_config_for_shortcut(obj, config_str):
    tokens = config_str.split(':')
    if len(tokens) < 2:
      return None
    language_model = tokens[0]
    language_model_dataset = tokens[1]
    supervised_finetune = None
    supervised_dataset = None
    if len(tokens) > 2:
      supervised_finetune = tokens[2]
      supervised_dataset = tokens[3]

    print('Language Model: ' + language_model + ' using dataset at ' + language_model_dataset)
    if supervised_finetune is not None:
      print('Supervised Model: ' + supervised_finetune + ' using dataset at ' + supervised_dataset)
    
    if supervised_finetune is not None:
      # Case of finetuning
      template_string = language_model + '+best2010'
      default_template_path = ConfigMapper.get_config_path_for(template_string)
    else:
      # Case of language model
      default_template_path = ConfigMapper.get_config_path_for(language_model)
      config = None
      with open(default_template_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

      # Replace data set source
      config['dataset']['class'] = 'LocalDatasetWrapper'
      config['dataset']['config']['dataset_name'] = language_model_dataset.replace('/', '_').replace('\\', '_')
      config['dataset']['config']['local_src_dir'] = language_model_dataset

      # Replace output directory
      config['execution']['config']['output_dir'] = '_outputs_/' + language_model_dataset.replace('/', '_').replace('\\', '_')

      print(config['dataset'])      
      print(config['execution'])      
      return config

    return None

ConfigMapper.get_config_path_for = classmethod(ConfigMapper.get_config_path_for)
ConfigMapper.construct_json_config_for_shortcut = classmethod(ConfigMapper.construct_json_config_for_shortcut)
