import gin

import argparse

gin.finalize()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gin', type=str, required=True)
  parser.add_argument('--eval_steps', type=int, default=49999)
  parser.add_argument('--eval_bs', type=int, default=1)
  parser.add_argument('--steps', type=int, default=0)

  args = parser.parse_args()
  with gin.unlock_config():
    gin.parse_config_file(args.gin)

    n_steps = args.steps or gin.query_parameter('train.steps')
    gin.bind_parameter('train.steps', n_steps + 1)

    gin.bind_parameter('batcher.eval_batch_size', args.eval_bs)
    gin.bind_parameter('train.eval_steps', args.eval_steps)
    gin.bind_parameter('train.eval_frequency', 1)

    gin.bind_parameter('train.optimizer',
                       gin.config.ConfigurableReference('trax.optimizers.SGD',
                                                        False))
    gin.bind_parameter('train.lr_schedule_fn',
                       gin.config.ConfigurableReference(
                           'trax.supervised.lr_schedules.constant', False))
    gin.bind_parameter('trax.supervised.lr_schedules.constant.value', 0.0)

  *filename, _ = args.gin.split('.')
  eval_filename = '.'.join(filename) + '_eval.gin'

  with open(eval_filename, 'w') as f:
    f.write(gin.config_str())
