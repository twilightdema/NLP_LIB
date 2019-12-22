NLP_LIB
=======
The python library for language modeling and fine tuning using Transformer based deep learning models with built-in Thai data set supported.

### Features
#### Lanugage Models Supported
 - Transformer Decoder-only model (Next token predicton objective function)
 - Transformer Encoder-only model (Masked tokens prediction objective function)
#### Fine Tuning Models Supported
 - Sequence-to-Sequence Model
 - Multi Class Classification
 - Multi Label Classification
#### Built-in Data Set Supported
 - **NECTEC BEST2010** for Language Model
 - **Thailand Wikipedia Dump** for Langauge Model
 - **NECTEC BEST2010** for Topic Classification
 - **Truevoice** for Intention Detection
 - **Wisesight** for Sentiment Analysis
 - **Wongnai** for Rating Prediction
#### Build-in Input / Output Transformation
 - Full word dictionary
 - Bi-gram dictionary
 - Sentencepiece dictionary
#### Other Features
 - Automatic multi-GPUs detection and training support (Data Parallel)
 - Automatic state saving and resume training
 - Automatic saving best model and last model during training
 - Automatic generate Tensorboard log
 - Sequence generation from language model using ARGMAX, BEAM Search
 - Support initialzation from trained language model weights in fine tuning
 - Modularized and fully extensible


Library Usages
==============
```
python3 nlp_core/engine.py <model_name | model_json_path> <operation> <extra_params>
```
 - **model_name**: Predefined model name shipped with the library (See appendix A. for list of predefined models)
 - **model_json_path**: JSON Configuration File path of the model (See appendix B. JSON file format)
 - **operation**: train | predict | generate - default is train (See example section for how to use "generate" mode)


Examples of running the training process
========================================

Train 6 layers of transformer decoder-only model
```
python3 nlp_core/engine.py tf6-dec
```
Finetune 4 layers of transformer encoder-only with sentencepiece dict model on truevoice data
```
python3 nlp_core/engine.py tf4-enc-sp+truevoice
```
Run prediction on input data file
```
python3 nlp_core/engine.py tf4-enc-sp+truevoice predict file:input_data.txt
```
Run prediction on input string
```
python3 nlp_core/engine.py tf4-dec-bigram+best2010 predict str:This,is,input,text
```
Run sequence generation for 20 tokens using BEAM search on 3 best prediction sequences
```
python3 nlp_core/engine.py tf6-dec generate:20:beam3 str:This,is,seed,text
```

APPENDIX A) List of predefined models
=====================================

#### For language model:
```
tf<N>-<Arch>-<Dict> : Transformer models
```
 - **N** : Number of transformer layers, support 2, 4, 6 and 16. Default is 6.
 - **Arch**: Architecture of language model, support "enc" and "dec" for encoder-only and decoder-only.
 - **Dict**: Data transformation, support "full", "bigram" and "sp" for full word dict, bigram dict and sentencepiece dict. Default is "full"

 **Examples**:
 tf-dec
 tf6-dec
 tf4-enc-full
 tf12-dec-sp
 tf2-enc-bigram

#### For fine tuning model:
```
tf<N>-<Arch>-<Dict>+<Finetune Data> : Transformer models
```
 - **N** : Number of transformer layers, support 2, 4, 6 and 16. Default is 6.
 - **Arch**: Architecture of language model, support "enc" and "dec" for encoder-only and decoder-only.
 - **Dict**: Data transformation, support "full", "bigram" and "sp" for full word dict, bigram dict and sentencepiece dict. Default is "full"
 - **Finetune Data**: Fine tuning data set, support "best2010", "truevoice", "wongnai" and "wisesight"

 **Examples**:
 tf-dec+best2010
 tf6-dec+truevoice
 tf4-enc-full+wongnai
 tf12-dec-sp+wisesight


APPENDIX B) JSON Configuration File Format
==========================================

This file defines how to run the model training.
The model training run is defined by 5 components below:
 - **Model** : Model architecture to be used
 - **Dataset** : Dataset to be used
 - **Input / Output Transformation** : How to encode / decode input and output data
 - **Callbacks** : List of additional flow need to be run in training loop
 - **Execution** : Training processes to be used, for example what optimizer, how many epoch

The JSON file needs to supply configuration of each component, the overall format is shown below:
```
{
  "model": {
    "class": <CLASS NAME OF MODEL>,
    "config": {
      <CONFIGURATIONS OF THE MODEL>
    }
  },
  "dataset": {
    "class": <CLASS NAME OF DATA SET>,
    "config": {
      <CONFIGURATIONS OF THE DATA SET>
    }
  },
  "input_transform": {
    "class": <CLASS NAME OF INPUT TRANSFORMATION>,
    "config": {
      <CONFIGURATIONS OF THE INPUT TRANFORMATION>
    }
  },
  "output_transform": {
    "class": <CLASS NAME OF OUTPUT TRANSFORMATION>,
    "config": {
      <CONFIGURATIONS OF THE OUTPUT TRANFORMATION>
    }
  },
  "callbacks": [
    .... [MULTIPLE CALLBACK HOOKS] ....
    {
      "class": <CLASS NAME OF CALLBACK HOOKS>,
      "config": {
        <CONFIGURATIONS OF THE CALLBACK>
      }
    }
  ],
  "execution": {
    "config": {
      <CONFIGURATIONS OF THE TRAINING PROCESS>
    }
  }
}
```
Overall is that the configuration of each module requires class name of the module and also
configurations for them. The required / optional configurations of each module are depended
on module class so you have to read document for each module class to find out how to config them.
The class name of each module is used to look up for implementation of the module in 
the following directories:

 - **model** => ./models
 - **dataset** => ./datasets
 - **input / output transformations** => ./transforms
 - **callbacks** => ./callbacks

You can implement new module by putting module python class in above directories and the library
will be able to resolve for implementation when it finds class name in JSON configuration file.

Below is example of JSON configuration file for training 12 layers of transformer decoder-only model
with sentencepiece dictionary data transformation and dynamic learning rate on THWIKI data set:
```
{
  "model": {
    "class": "TransformerDecoderOnlyWrapper",
    "config": {
      "len_limit": 256,
      "d_model": 512,
      "d_inner_hid": 2048,
      "n_head": 8,
      "d_k": 512,
      "d_v": 512,
      "layers": 12,
      "dropout": 0.1,
      "share_word_emb": true,
      "max_input_length": 256,
      "cached_data_dir": "_cache_"
    }
  },
  "dataset": {
    "class": "THWIKILMDatasetWrapper",
    "config": {
      "base_data_dir": "_tmp_"
    }
  },
  "input_transform": {
    "class": "SentencePieceDictionaryWrapper",
    "config": {
      "column_id": 0,
      "max_dict_size" : 15000,
      "mask_last_token": false
    }
  },
  "output_transform": {
    "class": "SentencePieceDictionaryWrapper",
    "config": {
      "column_id": 1,
      "max_dict_size" : 15000
    }
  },
  "callbacks": [
    {
      "class": "DynamicLearningRateWrapper",
      "config": {
        "d_model": 512,
        "warmup": 50000,
        "scale": 0.5
      }
    }
  ],
  "execution": {
    "config": {
      "optimizer": "adam",
      "optimizer_params": [0.1, 0.9, 0.997, 1e-9],
      "batch_size": 32,
      "epochs": 60,
      "watch_metric": "val_acc",
      "output_dir": "_outputs_/thwikilm_tfbase12_dec_s2_sp",
      "save_weight_history": false,
      "resume_if_possible": true,
      "multi_gpu": false
    }
  }
}
```
Below is another example of using the trained model above to finetune on TRUEVOICE data set.
Note that we use "SequenceTransferLearningWrapper" model class, which accept configuration of
language model to be used as an encoder and also the original data set configuration used to pre-train
the encoder model:
```
{
  "model": {
    "class": "SequenceTransferLearningWrapper",
    "config": {
      "output_class_num": 8,
      "encoder_checkpoint": "_outputs_/thwikilm_tfbase12_dec_s2_sp/checkpoint/best_weight.h5",
      "train_encoder": true,
      "max_input_length": 256,
      "drop_out": 0.4,
      "cached_data_dir": "_cache_",

      "encoder_model": {
        "class": "TransformerDecoderOnlyWrapper",
        "config": {
          "len_limit": 256,
          "d_model": 512,
          "d_inner_hid": 2048,
          "n_head": 8,
          "d_k": 512,
          "d_v": 512,
          "layers": 12,
          "dropout": 0.1,
          "share_word_emb": true,
          "max_input_length": 256,
          "cached_data_dir": "_cache_"
        }
      },

      "encoder_dict_dataset": {
        "class": "THWIKILMDatasetWrapper",
        "config": {
          "base_data_dir": "_tmp_"
        }
      }
    
    }
  },
  "dataset": {
    "class": "TruevoiceDatasetWrapper",
    "config": {
      "base_data_dir": "_tmp_"
    }
  },
  "input_transform": {
    "class": "SentencePieceDictionaryWrapper",
    "config": {
      "column_id": 0,
      "max_dict_size" : 15000,
      "clf_pos_offset": -1,
      "clf_id": 3,
      "mask_last_token": false
    }
  },  
  "output_transform": {
    "class": "SingleClassTransformWrapper",
    "config": {
      "column_id": 1
    }
  },
  "callbacks": [
    {
      "class": "DynamicLearningRateWrapper",
      "config": {
        "d_model": 512,
        "warmup": 50000,
        "scale" : 0.25
      }
    }
  ],
  "execution": {
    "config": {
      "optimizer": "adam",
      "optimizer_params": [0.0001, 0.9, 0.98, 1e-9],
      "batch_size": 32,
      "epochs": 30,
      "watch_metric": "val_acc",
      "output_dir": "_outputs_/finetune_thwikilm_tfbase12_dec_s2s_sp_truevoice",
      "save_weight_history": false,
      "resume_if_possible": true
    }
  }
}
```