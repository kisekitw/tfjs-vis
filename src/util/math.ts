/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {max, maximum, scalar, Tensor, Tensor1D, tidy} from '@tensorflow/tfjs';

import {ConfusionMatrixData, HistogramStats} from '../types';

import {assert} from './utils';

/**
 * Returns summary statistics for an array of numbers
 *
 * @param input
 */
export function arrayStats(input: number[]): HistogramStats {
  if (!Array.isArray(input)) {
    throw new Error('input must be an array');
  }
  if (input.length === 0) {
    return {
      numVals: 0,
      numNans: 0,
      numZeros: 0,
      max: undefined,
      min: undefined,
    };
  }

  const numVals = input.length;
  let max = -Infinity;
  let min = Infinity;
  let numZeros = 0;
  let numNans = 0;

  for (let i = 0; i < numVals; i++) {
    const curr = input[i];
    if (curr > max) {
      max = curr;
    }

    if (curr < min) {
      min = curr;
    }

    if (curr === 0) {
      numZeros += 1;
    }

    if (isNaN(curr)) {
      numNans += 1;
    }
  }

  const result = {
    numVals,
    numZeros,
    numNans,
    max,
    min,
  };

  // Handle all NaN input
  if (result.max === -Infinity) {
    result.max = NaN;
  }
  if (result.min === Infinity) {
    result.min = NaN;
  }

  return result;
}

/**
 * Returns summary statistics for a numeric tensor. *
 *
 * @param input
 */
export async function tensorStats(input: Tensor): Promise<HistogramStats> {
  // TODO. Benchmark this and consider having one of the *stats functions
  // delegate to the other.

  const [min, max, numZeros] = tidy(() => {
    const zero = scalar(0, input.dtype);

    const min = input.min();
    const max = input.max();
    const numZeros = input.equal(zero).sum();

    return [min, max, numZeros];
  });

  return await Promise
      .all([input.data(), min.data(), max.data(), numZeros.data()])
      .then(([tensorVal, minVal, maxVal, numZerosVal]) => {
        // We currently need to count NaNs on CPU.
        const numVals = tensorVal.length;
        let numNans = 0;
        for (let i = 0; i < numVals; i++) {
          const curr = tensorVal[i];
          if (isNaN(curr)) {
            numNans += 1;
          }
        }

        let trueMin = minVal[0];
        let trueMax = maxVal[0];
        if (numNans === numVals) {
          // on gpu the min and max won't be accurate if all values are NaN
          trueMin = NaN;
          trueMax = NaN;
        }

        const stats = {
          numVals,
          numZeros: numZerosVal[0],
          numNans,
          min: trueMin,
          max: trueMax,
        };

        return stats;
      });
}

/**
 *
 * @param labels
 * @param predictions
 * @param numClasses
 */
export async function confusionMatrix(
    labels: Tensor1D, predictions: Tensor1D,
    numClasses?: number): Promise<number[][]> {
  assert(labels.rank === 1, 'labels must be a 1D tensor');
  assert(predictions.rank === 1, 'predictions must be a 1D tensor');
  assert(
      labels.size === predictions.size,
      'labels and predictions must be the same length');

  let numClasses_: number;
  if (numClasses == null) {
    numClasses_ = tidy(() => {
      return maximum(labels.max(), predictions.max()).dataSync()[0] + 1;
    });
  } else {
    numClasses_ = numClasses;
  }

  return Promise.all([labels.data(), predictions.data()])
      .then(([labelsArray, predsArray]) => {
        const result: number[][] = Array(numClasses_).fill(0);
        // Initialize the matrix
        for (let i = 0; i < numClasses_; i++) {
          result[i] = Array(numClasses_).fill(0);
        }

        for (let i = 0; i < labelsArray.length; i++) {
          const labelIndex = labelsArray[i];
          const predIndex = predsArray[i];

          result[labelIndex][predIndex] += 1;
        }

        return result;
      });
}
