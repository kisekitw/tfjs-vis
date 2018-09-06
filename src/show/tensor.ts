import {Tensor} from '@tensorflow/tfjs';

import {renderHistogram} from '../render/histogram';
import {getDrawArea} from '../render/render_utils';
import {Drawable} from '../types';
import {tensorStats} from '../util/math';

/**
 * Shows a histogram with the distribution of all values in a given tensor.
 *
 * @param container A `{name: string, tab?: string}` object specifying which
 *  surface to render to.
 * @param tensor the input tensor
 */
export async function distribution(container: Drawable, tensor: Tensor) {
  const drawArea = getDrawArea(container);
  const stats = await tensorStats(tensor);
  const values = await tensor.data();
  renderHistogram(values, drawArea, {height: 150, stats});
}
