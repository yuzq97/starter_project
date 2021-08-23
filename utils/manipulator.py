"""Utility functions for latent codes manipulation."""

import numpy as np
import argparse
import os.path

__all__ = ['project_boundary', 'linear_interpolate']

def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.

  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].
  
  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.
  
  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.
  
  Raises:
    LinAlgError: If there are more than two condition boundaries and the method fails 
                 to find a projected boundary orthogonal to all condition boundaries.
  """
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal
  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)
  elif len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)
  else:
    for cond_boundary in args:
      assert (len(cond_boundary.shape) == 2 and cond_boundary.shape[0] == 1 and
              cond_boundary.shape[1] == primal.shape[1])
    cond_boundaries = np.squeeze(np.asarray(args))
    A = np.matmul(cond_boundaries, cond_boundaries.T)
    B = np.matmul(cond_boundaries, primal.T)
    x = np.linalg.solve(A, B)
    new = primal - (np.matmul(x.T, cond_boundaries))
    return new / np.linalg.norm(new)


def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
  """Manipulates the given latent code with respect to a particular boundary.

  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.

  NOTE: Distance is sign sensitive.

  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] \n'
                   f'But {latent_code.shape} is received.')


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-p', '--primal', type=str, required=True,
                      help='Path to the primal boundary. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-f', '--first_condition', type=str,
                      help='Path to the first condition boundary.')
  parser.add_argument('-s', '--second_condition', type=str,
                      help='Path to the second condition boundary.')

  return parser.parse_args()

if __name__=='__main__':
  args = parse_args()
  if args.first_condition:
    primal = np.load(args.primal)
    if args.second_condition:
      first = np.load(args.first_condition)
      second = np.load(args.second_condition)
      projected = project_boundary(primal, first, second)
    else:
      first = np.load(args.first_condition)
      projected = project_boundary(primal, first)
      np.save(os.path.join(args.output_dir, 'boundary.npy'), projected)
  else:
    print('There should be at least one condition boundary!')

