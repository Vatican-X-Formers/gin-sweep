import os
import shutil
from pathlib import Path
from typing import Iterator

import gin
import trax

from gin_sweep import exp_name_from_params
from gin_sweep.sweep import config_overrides_from_sweep

gin.finalize()  # Do not allow unexpected ginfile changes


def load_base_gin(gin_repo_path,
                  configs_root='/content/vatican-gins/configs/') -> str:
    with open(configs_root + gin_repo_path) as f:
        base_ginfile = f.read()
    return base_ginfile


class ExperimentInstanceConfig:
    def __init__(self, base_gin_path: str,
                 sweep_override: dict):
        self.base_gin_path = base_gin_path
        self.sweep_override = sweep_override

    def __str__(self):
        repo_prefix = self._repo_path_prefix()
        sweep_suffix = exp_name_from_params(self.sweep_override)
        return f'{repo_prefix}@{sweep_suffix}'

    def _repo_path_prefix(self):
        cfg_dir, gin_cfg_name = self.base_gin_path.split('/')
        cfg_name, _ = gin_cfg_name.split('.')
        return f'{cfg_dir}#{cfg_name}'


class ExperimentConfig:
    def __init__(self, trax_branch_name, t2t_branch_name, base_gin_path,
                 sweep: dict):
        self.trax_branch_name = trax_branch_name
        self.t2t_branch_name = t2t_branch_name
        self.base_gin_path = base_gin_path
        self.sweep = sweep

    def generate_experiment_instances(self) -> \
            Iterator[ExperimentInstanceConfig]:
        overrides = config_overrides_from_sweep(self.sweep)
        return (
            ExperimentInstanceConfig(
                self.base_gin_path,
                sweep_override
            ) for sweep_override in overrides
        )


class ExperimentSaver:
    def __init__(self, saved_models_dir: str, trax_train_dir: str = None):
        if trax_train_dir is None:
            trax_train_dir = os.path.expanduser('~/train_dir/')
        self.output_dir = trax_train_dir
        self.saved_models_path = Path(saved_models_dir)
        assert self.saved_models_path.is_dir(), \
            "Saved models destination is not a directory"

    def clear_train_dir(self):
        shutil.rmtree(self.output_dir + '/*')

    def save_as(self, cfg: ExperimentConfig,
                instance_cfg: ExperimentInstanceConfig):
        exp_name = str(instance_cfg)
        branch_name = cfg.trax_branch_name

        dest_dir = self._get_destination_dir(branch_name, exp_name)
        shutil.move(self.output_dir, dest_dir)

    def _get_destination_dir(self, branch_name, exp_name):
        return str(self.saved_models_path / branch_name / exp_name)


class ExperimentRun:
    def __init__(self, cfg: ExperimentConfig,
                 saver: ExperimentSaver):
        self.cfg = cfg
        self.saver = saver
        self.base_gin = load_base_gin(self.cfg.base_gin_path)

    def run(self, save=True, model=None):
        for instance_cfg in self.cfg.generate_experiment_instances():
            self.run_instance(instance_cfg, save, model)

    def run_instance(self, instance_cfg: ExperimentInstanceConfig, save: bool,
                     model=None):
        self.saver.clear_train_dir()

        self._train(instance_cfg, model)

        if save:
            self.saver.save_as(self.cfg, instance_cfg)

    def _train(self, instance_cfg: ExperimentInstanceConfig,
               verbose=True, model: str = None):
        with gin.unlock_config():
            gin.parse_config(self.base_gin)

            if verbose:
                print(self.base_gin)

            for k, v in instance_cfg.sweep_override.items():
                gin.bind_parameter(k, v)

            if model is not None:
                gin.parse_config(f"train.model = @trax.models.{model}")

            print(gin.config_str())
            train = trax.supervised.train(output_dir=self.saver.output_dir)
            return train
