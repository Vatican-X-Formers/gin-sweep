import unittest

import yaml

from gin_sweep import gin_configs_from_yaml


class MyTestCase(unittest.TestCase):
    def test_sweep(self):
        test_yaml = 'bert_glue_sweep.yaml'
        test_ginfile = 'bert_glue.gin'

        with (
                open(test_yaml)
        ) as yaml_config, (
                open(test_ginfile)
        ) as gin_config_file:
            sweep = yaml.load(yaml_config, Loader=yaml.FullLoader)
            gin = gin_config_file.read()

        for cfg in gin_configs_from_yaml(gin, sweep, 30):
            self.assertGreater(len(cfg), 0, 'gin should not be empty')


if __name__ == '__main__':
    unittest.main()
