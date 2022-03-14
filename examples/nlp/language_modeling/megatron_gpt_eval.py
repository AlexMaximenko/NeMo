# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState

assert torch.cuda.is_available()


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


@hydra_runner(config_path="conf", config_name="megatron_gpt_inference")
def main(cfg) -> None:
    # cast precision to int if 32 or 16
    if cfg.precision in ["32", "16"]:
        cfg.precision = int(float(cfg.precision))

    # trainer required for restoring model parallel models
    trainer = Trainer(
        plugins=NLPDDPPlugin(),
        devices=cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size,
        accelerator='gpu',
        precision=cfg.precision,
    )

    app_state = AppState()
    if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            _,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
        )

    model = MegatronGPTModel.restore_from(restore_path=cfg.model_file, trainer=trainer)
    model.freeze()

    ds = RequestDataSet(cfg.prompts)
    request_dl = DataLoader(dataset=ds, batch_size=2)

    # has to turn off activations_checkpoint_method for inference
    model.model.language_model.encoder.activations_checkpoint_method = None
    # set the inference config
    with open_dict(model.cfg):
        model.cfg.inference = cfg.inference

    response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
