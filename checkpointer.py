from llmfoundry.callbacks import HuggingFaceCheckpointer
from qllmt.nn import is_wrapped, wrap_model, unwrap_model, unpatch_fsdp_model

import contextlib
import copy
import logging
import os
import shutil
import tempfile
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any, Dict
import json

import torch
import torch.nn as nn
from composer.core import Callback, State
from composer.loggers import Logger
from composer.core.state import fsdp_state_dict_type_context
from composer.utils import (
    dist,
    format_name_with_dist_and_time,
)
from packaging import version
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmfoundry.models.mpt import MPTConfig, MPTForCausalLM
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility

log = logging.getLogger(__name__)

def find_hadamard_config(model):
    config = getattr(model, "hadamard_config", None)
    if config is not None:
        return config
    if hasattr(model, "module"):
        return find_hadamard_config(model.module)
    if hasattr(model, "model"):
        return find_hadamard_config(model.model)
    return None


class HadHFCheckpointer(HuggingFaceCheckpointer):
    def _save_checkpoint(self, state: State, logger: Logger):
        del logger  # unused

        self.last_checkpoint_batch = state.timestamp.batch

        log.info('Saving HuggingFace formatted checkpoint')

        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        CONFIG_MAPPING._extra_content['mpt'] = MPTConfig
        MPTConfig.register_for_auto_class()
        MPTForCausalLM.register_for_auto_class('AutoModelForCausalLM')

        save_dir = format_name_with_dist_and_time(
            str(
                Path(self.save_dir_format_str) /
                self.huggingface_folder_name_fstr,
            ),
            state.run_name,
            state.timestamp,
        )

        # Use a temporary directory if save_dir is remote.
        use_temp_dir = self.remote_ud is not None
        temp_save_dir = tempfile.mkdtemp() if use_temp_dir else save_dir

        log.debug('Gathering state dict')
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if state.is_model_ddp:
            composer_model = state.model.module
            original_model: PreTrainedModel = state.model.module.model
            state_dict_model = state.model.module.model
            original_tokenizer = state.model.module.tokenizer
        elif isinstance(state.model.model, FSDP):
            composer_model = state.model
            original_model: PreTrainedModel = state.model.model.module
            state_dict_model = state.model.model
            original_tokenizer = state.model.tokenizer
        else:
            composer_model = state.model
            original_model: PreTrainedModel = state.model.model
            state_dict_model = state.model.model
            original_tokenizer = state.model.tokenizer

        print('Checkpointing...')
        unpatch_fsdp_model(state_dict_model)

        if version.parse(torch.__version__) > version.parse('2.2.9'):
            from torch.distributed._tensor import DTensor
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )
            cpu_offload = True

            # Add a dtensor->cpu tensor hook to avoid CUDA OOM
            def dtensor_to_tensor_hook(
                module: nn.Module,
                state_dict: Dict[str, Any],
                prefix: str,
                *args: Any,
            ) -> Dict[str, Any]:
                dtensor_fqns = []
                for fqn in state_dict.keys():
                    tensor = state_dict[fqn]
                    if isinstance(tensor, DTensor):
                        dtensor_fqns.append(fqn)
                        tensor = tensor.full_tensor()  # type: ignore
                        if dist.get_global_rank() == 0:
                            if cpu_offload:
                                tensor = tensor.cpu()
                            state_dict[fqn] = tensor
                if dist.get_global_rank() != 0:
                    for fqn in dtensor_fqns:
                        del state_dict[fqn]
                return state_dict

            hooks = []
            for _, module in state_dict_model.named_modules():
                if isinstance(module, FSDP):
                    hooks.append(
                        module.
                        _register_state_dict_hook(dtensor_to_tensor_hook),
                    )

            state_dict = get_model_state_dict(
                state_dict_model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=cpu_offload,
                ),
            )
            for hook in hooks:
                hook.remove()
        else:
            state_dict_context = fsdp_state_dict_type_context(
                original_model,
                state_dict_type='full',
            ) if ((not state.is_model_ddp) and
                  isinstance(state_dict_model,
                             FSDP)) else contextlib.nullcontext()
            with state_dict_context:
                state_dict = state_dict_model.state_dict()

        # Convert the state dict to the requested precis
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.to(dtype=self.dtype)

        new_model_instance = None  # Need this for pyright because variable could be unbound

        if dist.get_global_rank() == 0:
            log.debug('Saving Hugging Face checkpoint in global rank 0')

            # Edit HF config before building 2nd model copy
            copied_config = copy.deepcopy(original_model.config)
            if copied_config.model_type == 'mpt':
                copied_config.attn_config['attn_impl'] = 'torch'
                copied_config.init_device = 'cpu'
                if 'moe_world_size' in getattr(copied_config, 'ffn_config', {}):
                    copied_config.ffn_config['moe_world_size'] = 1

            log.debug(f'Creating new model instance')

            if composer_model.using_peft:
                # We don't use meta here because the state dict does not contain the full
                # model, only the adapter weights.
                active_adapter = original_model.active_adapter
                base_model = original_model.get_base_model()
                new_base_model_instance = type(base_model)(copied_config)

                new_model_instance = type(original_model)(
                    new_base_model_instance,
                    original_model.peft_config[active_adapter],
                )
                new_model_instance.to(dtype=self.dtype)
            else:
                # First create the model instance on meta device to avoid the
                # initialization cost.
                with init_empty_weights():
                    new_model_instance = type(original_model)(copied_config)
                    new_model_instance.generation_config.update(
                        **original_model.generation_config.to_dict(),
                    )

            # Then load the state dict in with "assign" so that the state dict
            # is loaded properly even though the model is initially on meta device.

            hadamard_config = find_hadamard_config(state.model)
            if is_wrapped(original_model):
                wrap_model(new_model_instance, config=hadamard_config)

            new_model_instance.load_state_dict(state_dict, assign=True)
            del state_dict

            if is_wrapped(new_model_instance):
                unwrap_model(new_model_instance)

            # Transform the model and tokenizer before saving
            new_model_instance, original_tokenizer = self.transform_model_and_tokenizer(
                new_model_instance,
                original_tokenizer,
            )

            log.debug('Saving Hugging Face checkpoint to disk')
            new_model_instance.save_pretrained(temp_save_dir)
            if original_tokenizer is not None:
                assert isinstance(original_tokenizer, PreTrainedTokenizerBase)
                original_tokenizer.save_pretrained(temp_save_dir)
            
            if hadamard_config is not None:
                hadamard_config_path = os.path.join(temp_save_dir, 'hadamard_config.json')
                with open(hadamard_config_path, 'w') as f:
                    json.dump(hadamard_config, f)

            # Only need to edit files for MPT because it has custom code
            if original_model.config.model_type == 'mpt':
                log.debug('Editing MPT files for HuggingFace compatibility')
                edit_files_for_hf_compatibility(
                    temp_save_dir,
                    self.flatten_imports,
                )

            if self.remote_ud is not None:
                for filename in os.listdir(temp_save_dir):
                    remote_file_name = os.path.join(save_dir, filename)
                    remote_file_uri = self.remote_ud.remote_backend.get_uri(
                        remote_file_name,
                    )
                    log.info(
                        f'Uploading HuggingFace formatted checkpoint to {remote_file_uri}',
                    )
                    self.remote_ud.upload_file(
                        state=state,
                        remote_file_name=remote_file_name,
                        file_path=Path(os.path.join(temp_save_dir, filename)),
                        overwrite=self.overwrite,
                    )

        dist.barrier()

        if dist.get_global_rank() == 0:
            if self.mlflow_registered_model_name and self._is_last_batch(state):
                components = {'model': new_model_instance}
                if original_tokenizer is not None:
                    components['tokenizer'] = original_tokenizer

                log.debug('Logging Hugging Face model to MLFlow')
                for i, mlflow_logger in enumerate(self.mlflow_loggers):
                    log.debug(
                        f'Registering model to UC at {mlflow_logger.model_registry_prefix}.{self.mlflow_registered_model_name}',
                    )
                    local_save_path = str(
                        Path(temp_save_dir) / f'mlflow_save_{i}',
                    )

                    # TODO: Remove after mlflow fixes the bug that makes this necessary
                    import mlflow
                    mlflow.store._unity_catalog.registry.rest_store.get_feature_dependencies = lambda *args, **kwargs: ''
                    model_saving_kwargs: Dict[str, Any] = {
                        'path': local_save_path,
                    }
                    if composer_model.using_peft:
                        model_saving_kwargs['flavor'] = 'peft'
                        model_saving_kwargs['save_pretrained_dir'
                                           ] = temp_save_dir
                        model_saving_kwargs[
                            'metadata'] = self.mlflow_logging_config['metadata']
                    else:
                        model_saving_kwargs['flavor'] = 'transformers'
                        model_saving_kwargs['transformers_model'] = components
                        model_saving_kwargs.update(self.mlflow_logging_config)

                    mlflow_logger.save_model(**model_saving_kwargs)

                    # Upload the license file generated by mlflow during the model saving.
                    license_filename = _maybe_get_license_filename(
                        local_save_path,
                        self.mlflow_logging_config['metadata'].get(
                            'pretrained_model_name',
                            None,
                        ),
                    )
                    if license_filename is not None:
                        mlflow_logger._mlflow_client.log_artifact(
                            mlflow_logger._run_id,
                            os.path.join(local_save_path, license_filename),
                        )

                    # Spawn a new process to register the model.
                    process = SpawnProcess(
                        target=_register_model_with_run_id_multiprocess,
                        kwargs={
                            'mlflow_logger':
                                mlflow_logger,
                            'composer_logging_level':
                                logging.getLogger('composer').level,
                            'model_uri':
                                local_save_path,
                            'name':
                                self.mlflow_registered_model_name,
                            'await_creation_for':
                                3600,
                        },
                    )
                    process.start()
                    self.child_processes.append(process)

                    # Save the temporary directory to be cleaned up later.
                    if use_temp_dir:
                        self.temp_save_dir = temp_save_dir
            else:
                # Clean up the temporary directory if we don't need to register to mlflow.
                if use_temp_dir:
                    shutil.rmtree(temp_save_dir)
        dist.barrier()

def wrap_from_pretrained(model, path):
    hadamard_config = None
    hadamard_config_path = os.path.join(path, 'hadamard_config.json')
    if os.path.exists(hadamard_config_path):
        with open(hadamard_config_path, 'r') as f:
            hadamard_config = json.load(f)

    if hadamard_config is not None:
        print(f'Wrapping model with config: {hadamard_config}')
        wrap_model(model, config=hadamard_config)
    else:
        print(f'No hadamard config found at {hadamard_config_path}')

    return model    