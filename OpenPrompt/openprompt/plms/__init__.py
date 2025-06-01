import itertools
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union, List, Tuple, Dict, Optional
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.tokenization_utils import PreTrainedTokenizer
from collections import defaultdict
from openprompt.utils import round_list
import numpy as np

from statistics import mode
from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from .utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from .mlm import MLMTokenizerWrapper
from .seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
from .lm import LMTokenizerWrapper
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, AlbertForMaskedLM, \
                         T5Config, T5Tokenizer, T5ForConditionalGeneration, \
                         OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
                         GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, \
                         OPTConfig, OPTForCausalLM, \
                         ElectraConfig, ElectraForMaskedLM, ElectraTokenizer, \
                         GPTJConfig, GPTJForCausalLM, \
                         XLNetTokenizerFast, XLNetConfig, XLNetLMHeadModel, \
                         XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer
from collections import namedtuple
from yacs.config import CfgNode

from kobert_tokenizer import KoBERTTokenizer

from openprompt.utils.logging import logger


ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    'kobert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': KoBERTTokenizer,
        'model': BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'xlmroberta' : ModelClass(**{
        'config': XLMRobertaConfig,
        'tokenizer': XLMRobertaTokenizer,
        'model':XLMRobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    't5':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5TokenizerWrapper
    }),
    't5-lm':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5LMTokenizerWrapper,
    }),
    'opt': ModelClass(**{
        'config': OPTConfig,
        'tokenizer': GPT2Tokenizer,
        'model': OPTForCausalLM,
        'wrapper': LMTokenizerWrapper,
    }),
    'electra': ModelClass(**{
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'model': ElectraForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    "gptj": ModelClass(**{
        "config": GPTJConfig, 
        "tokenizer": GPT2Tokenizer, 
        "model": GPTJForCausalLM,
        "wrapper": LMTokenizerWrapper
    }),
}


def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]


def load_plm(model_name, model_path, specials_to_add = None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path)
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in model_name: # add pad token for gpt
        specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = model_class.model.from_pretrained(model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    if 'opt' in model_name:
        tokenizer.add_bos_token=False
    return model, tokenizer, model_config, wrapper




def load_plm_from_config(config: CfgNode):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`model_config`: The wrapper class of this plm.
    """
    plm_config = config.plm
    model_class = get_model_class(plm_type = plm_config.model_name)
    model_config = model_class.config.from_pretrained(plm_config.model_path)
    # you can change huggingface model_config here
    # if 't5'  in plm_config.model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in plm_config.model_name: # add pad token for gpt
        if "<pad>" not in config.plm.specials_to_add:
            config.plm.specials_to_add.append("<pad>")
    model = model_class.model.from_pretrained(plm_config.model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(plm_config.model_path)
    wrapper = model_class.wrapper
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=config.plm.specials_to_add)
    return model, tokenizer, model_config, wrapper

def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer.

    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.

    Returns:
        The resized model, The tokenizer with the added special tokens.

    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                logger.info("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer




class TokenizerWrapper:
    def __init__(self,
                 max_seq_length: int,
                 tokenizer: PreTrainedTokenizer,
                 truncate_method: Optional[str] = 'tail',
                 create_token_type_ids: Optional[str] = False,
                 **kwargs):
        self.max_seq_length = max_seq_length

        self.tokenizer = tokenizer
        if truncate_method=='tail':
            self.truncate_fct = self.truncate_from_tail
        elif truncate_method=='head':
            self.truncate_fct = self.truncate_from_head
        elif truncate_method == 'balanced':
            self.truncate_fct = self.balanced_truncate
        else:
            raise NotImplementedError

        self.create_token_type_ids = create_token_type_ids

        self.template_mask_token = '<mask>'
        self.template_eos_token = '<eos>'
        self.template_bos_token = '<bos>'
        self.template_sep_token = '<sep>'
        self.template_cls_token = '<cls>'
        self.template_pad_token = '<pad>'

        from transformers import logging
        verbosity_before = logging.get_verbosity()
        logging.set_verbosity(logging.CRITICAL) # TODO solve this in a more elegant way
        self.mask_token_map = {self.template_mask_token: self.tokenizer.mask_token if hasattr(self.tokenizer, 'mask_token') else ''}
        self.eos_token_map = {self.template_eos_token: self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') else ''}
        self.bos_token_map = {self.template_bos_token: self.tokenizer.bos_token if hasattr(self.tokenizer, 'bos_token') else ''}
        self.sep_token_map = {self.template_sep_token: self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else ''}
        self.cls_token_map = {self.template_cls_token: self.tokenizer.cls_token if hasattr(self.tokenizer, 'cls_token') else ''}
        self.pad_token_map = {self.template_pad_token: self.tokenizer.pad_token if hasattr(self.tokenizer, 'pad_token') else ''}
        logging.set_verbosity(verbosity_before)

        self.num_truncated_sentences = 0
        self.total_passed_sentences = 0

    @property
    def truncate_rate(self,):
        r"""Using this function, one can easily identify how many sentence has be truncated, thus help the user to choose a better thresthold for chunking.
        """
        if self.total_passed_sentences==0:
            return None
        else:
            return self.num_truncated_sentences/self.total_passed_sentences

    @property
    def special_tokens_maps(self,) -> Dict:
        r"""This need to be specified in specific language model
        """
        if not hasattr(self, "_special_tokens_map"):
            _special_tokens_map = {}
            for attrname in self.__dict__.keys():
                if attrname.endswith('_token_map'):
                    _special_tokens_map.update(getattr(self, attrname))
        return  _special_tokens_map

    def tokenize_with_mask(self,
                            wrapped_example: List[Dict],
                            ) -> InputFeatures:
        raise NotImplementedError

    def tokenize_without_mask(self,
                            wrapped_example: List[Dict],
                            ) -> InputFeatures:
        raise NotImplementedError

    @staticmethod
    def balanced_truncate(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        '''truncate the inputs with balance, number of cut tokens is proportional to the part's length.
        '''
        shortenable_lens = [len(parts) if parts[0]==1 else 0
                                  for parts in input_dict['shortenable_ids']]
        total_shortenable_len = sum(shortenable_lens)
        num_tokens_to_truncate_each_part = [part_len/total_shortenable_len*num_tokens_to_truncate
                                                for part_len in shortenable_lens]
        round_list(num_tokens_to_truncate_each_part, num_tokens_to_truncate)

        truncated_example = defaultdict(list)
        for key in input_dict:
            parts = input_dict[key]
            for num_tokens_to_truncate_part, part in zip(num_tokens_to_truncate_each_part, parts):
                truncated_example[key].append(part[:len(part)-num_tokens_to_truncate_part])
        return truncated_example

    @staticmethod
    def truncate_from_tail(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the rear
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts[::-1]):
                if len(part) == 0: # to prevent some part are empty after tokenization
                    continue
                if shortenable_ids[-1-i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[-1-i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def truncate_from_head(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the head
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts):
                if shortenable_ids[i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def concate_parts(input_dict: Dict) -> Dict:
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict

    @staticmethod
    def padding(input_dict: Dict,
                max_len: int, pad_id_for_inputs: int=0, pad_id_for_others: int=0) -> None:
        for key, value in input_dict.items():
            if (len(input_dict[key]) > max_len):
                raise ValueError(f'''Truncated seq length of '{key}' still greater than max length {max_len}."\
                    "One possible reason is that no enough shortenable parts in template. Try adding {{"shortenable": "True"}} property.
                ''')
            if 'input' in key:
                input_dict[key].extend([pad_id_for_inputs]*(max_len-len(value)))
            else:
                input_dict[key].extend([pad_id_for_others]*(max_len-len(value)))
        return input_dict


    def add_special_tokens(self, encoder_inputs):
            # add special tokens
        for key in encoder_inputs:
            if key == "input_ids":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    encoder_inputs[key] = self.tokenizer.build_inputs_with_special_tokens(
                                                        encoder_inputs[key])
            else:
                special_tokens_mask = np.array(self.tokenizer.get_special_tokens_mask(encoder_inputs[key]))
                with_special_tokens = np.array(self.tokenizer.build_inputs_with_special_tokens(encoder_inputs[key]))
                if key in ["soft_token_ids"]: # TODO maybe more than this
                    encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens).tolist() # use 0 as special
                else:
                    encoder_inputs[key] =  ((1-special_tokens_mask) * with_special_tokens - special_tokens_mask*100).tolist() # use -100 as special
        return encoder_inputs

    def truncate(self, encoder_inputs):
        total_tokens = sum([len(part) for part in encoder_inputs['input_ids']])
        num_specials = self.num_special_tokens_to_add
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
        self.total_passed_sentences+=1
        if num_tokens_to_truncate>0:
            self.num_truncated_sentences += 1
            encoder_inputs = self.truncate_fct(input_dict=encoder_inputs,
                          num_tokens_to_truncate=num_tokens_to_truncate)
        return encoder_inputs