# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions to help with reproducibility of models. """

import os
import random
from typing import Optional

import numpy as np
import torch

from pytorch_lightning import _logger as log


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[int] = None) -> int:
    """Function that sets seed for pseudo-random number generators  in:
        pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable.
    """
    try:
        if seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
        else:
            seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        log.warning(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    seed = random.randint(min_seed_value, max_seed_value)
    log.warning(f"No correct seed found, seed set to {seed}")
    return seed


def get_seed_state() -> dict:
    """Function that collects seeds from pseudo-random number generators in:
        pytorch, numpy, python.random and reads PYTHONHASHSEED environment variable.
    """
    env_python_hash_seed = os.environ["PYTHONHASHSEED"]
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_state_all = torch.cuda.get_rng_state_all()

    state = {
        'PYTHONHASHSEED': env_python_hash_seed,
        'random_state': random_state,
        'numpy_state': numpy_state,
        'torch_state': torch_state,
        'torch_cuda_state_all': torch_cuda_state_all
    }

    return state


def set_seed_state(state: dict) -> None:
    """Function that sets seeds for pseudo-random number generators in:
        pytorch, numpy, python.random and writes PYTHONHASHSEED environment variable.
    """
    env_python_hash_seed = state['PYTHONHASHSEED']
    random_state = state['random_state']
    numpy_state = state['numpy_state']
    torch_state = state['torch_state']
    torch_cuda_state_all = state['torch_cuda_state_all']

    os.environ["PYTHONHASHSEED"] = env_python_hash_seed
    random.setstate(random_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state_all(torch_cuda_state_all)