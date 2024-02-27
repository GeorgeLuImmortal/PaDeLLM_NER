import torch

greedy_sample = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 512,
    "temperature": 1.0,
    "top_p": 1.0
    }

nucleus_sample = {
    "do_sample": True,
    "num_beams": 5,
    "max_new_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.9,
    }