from typing import List, Iterator, Tuple

import argparse
import itertools
import yaml
from sanitize_filename import sanitize


def config_overrides_from_yaml(yaml_sweep: dict) -> List[dict]:
    keys = yaml_sweep.keys()
    vals = yaml_sweep.values()
    return [dict(zip(keys, instance)) for instance in itertools.product(*vals)]


class GinWrapper:
    def __init__(self, ginfile: str):
        self._gin_lines = ginfile.splitlines()

    def override_param(self, key, value):
        line = filter(lambda x: x[1].strip().startswith(key),
                      enumerate(self._gin_lines))
        try:
            i, ln = next(line)
            self._gin_lines[i] = self._gin_line(key, value)
        except StopIteration:
            self._gin_lines.append(self._gin_line(key, value))

    @staticmethod
    def _gin_line(key, val) -> str:
        return f'{key} = {repr(val)}'

    def __str__(self):
        return '\n'.join(self._gin_lines)


def override_gin(base_ginfile: str, params_to_override: dict) -> str:
    gin_wrapper = GinWrapper(base_ginfile)
    for param, val in params_to_override.items():
        gin_wrapper.override_param(param, val)
    return str(gin_wrapper)


def gin_configs_from_yaml(base_ginfile: str, yaml_sweep: dict,
                          max_param_combinations: int) -> \
        Iterator[Tuple[dict, str]]:
    param_combinations = config_overrides_from_yaml(yaml_sweep)
    assert len(param_combinations) <= max_param_combinations, ValueError(
        f'Too many ginfile combinations, {len(param_combinations)}')

    return zip(param_combinations,
               map(lambda params: override_gin(base_ginfile, params),
                   param_combinations))


def exp_name_from_params(params: dict):
    return sanitize(
        '_'.join(f'{key[:10]}_{repr(val)}' for key, val in params.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', type=str, help='YAML sweep file path')
    parser.add_argument('gin_path', type=str, help='ginfile path')
    parser.add_argument('--max_n', type=int, default=30,
                        help='combinations limit')
    args = parser.parse_args()

    with (
            open(args.yaml_path)
    ) as yaml_config, (
            open(args.gin_path)
    ) as gin_config_file:
        sweep = yaml.load(yaml_config, Loader=yaml.FullLoader)
        gin = gin_config_file.read()

        for i, (_, cfg) in enumerate(
                gin_configs_from_yaml(gin, sweep, args.max_n)):
            with open(f'ginfile_{i}.gin', 'w+') as f:
                # TODO: gins named after experiment names?
                print(exp_name_from_params(_))
                f.write(cfg)
