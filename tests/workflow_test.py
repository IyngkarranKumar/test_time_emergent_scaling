import importlib
import pytest
import torch
import conf
import utils


importlib.reload(utils)
importlib.reload(conf)

from transformers import AutoTokenizer
from conf import config


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<bos>'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer

def test_count_generated_tokens_no_bos(tokenizer):
    sequence = torch.tensor([1, 2, 3, tokenizer.pad_token_id])
    question_length = 1
    assert utils.count_generated_tokens(sequence, tokenizer, question_length) == 2

def test_count_generated_tokens_with_bos(tokenizer):
    sequence = torch.tensor([tokenizer.pad_token_id, tokenizer.bos_token_id, 1, 2, 3])
    question_length = 1
    assert utils.count_generated_tokens(sequence, tokenizer, question_length) == 2

def test_count_generated_tokens_all_padding(tokenizer):
    sequence = torch.tensor([tokenizer.pad_token_id, tokenizer.pad_token_id])
    question_length = 1
    assert utils.count_generated_tokens(sequence, tokenizer, question_length) == 0

def test_count_generated_tokens_question_longer(tokenizer):
    sequence = torch.tensor([tokenizer.bos_token_id, 1, 2])
    question_length = 5
    assert utils.count_generated_tokens(sequence, tokenizer, question_length) == 0
