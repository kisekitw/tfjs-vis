import * as tf from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';

import {renderHistogram} from '../render/histogram';
import {getDrawArea} from '../render/render_utils';
import {renderTable} from '../render/table';
import {Drawable, HistogramStats} from '../types';
import {subSurface} from '../util/dom';
import {tensorStats} from '../util/math';

export async function modelSummary(container: Drawable, model: tf.Model) {
  const drawArea = getDrawArea(container);
  const summary = getModelSummary(model);

  const headers = [
    'Layer Name',
    'Output Shape',
    '# Of Params',
    'Trainable',
  ];

  const values = summary.layers.map(
      l =>
          [l.name,
           l.outputShape,
           l.parameters,
           l.trainable,
  ]);

  renderTable({headers, values}, drawArea);
}

export async function layer(container: Drawable, layer: Layer) {
  const drawArea = getDrawArea(container);
  const details = await getLayerDetails(layer);

  const headers = [
    'Tensor Name',
    'Shape',
    'Min',
    'Max',
    '# Zeros',
    '# NaNs',
  ];

  // Show layer summary
  const weightsInfoSurface = subSurface(drawArea, 'layer-weights-info');
  const detailValues = details.map(
      l =>
          [l.name, l.shape, l.stats.min, l.stats.max, l.stats.numZeros,
           l.stats.numNans]);
  renderTable({headers, values: detailValues}, weightsInfoSurface);

  // Show layer distribution
  const weights = await Promise.all(details.map(l => l.weight.data()));
  const values: number[] = [];
  for (const weight of weights) {
    for (let i = 0; i < weight.length; i++) {
      values.push(weight[i]);
    }
  }

  const layerValuesHistogram = subSurface(drawArea, 'param-distribution');
  renderHistogram(values, layerValuesHistogram, {height: 150, width: 460});
}

//
// Helper functions
//

function getModelSummary(model: tf.Model) {
  return {
    layers: model.layers.map(getLayerSummary),
  };
}

/*
 * Gets summary information/metadata about a layer.
 */
function getLayerSummary(layer: Layer) {
  let outputShape: string;
  if (Array.isArray(layer.outputShape[0])) {
    const shapes = (layer.outputShape as number[][]).map(s => formatShape(s));
    outputShape = `[${shapes.join(', ')}]`;
  } else {
    outputShape = formatShape(layer.outputShape as number[]);
  }

  return {
    name: layer.name,
    trainable: layer.trainable,
    parameters: layer.countParams(),
    outputShape,
  };
}

/*
 * Gets summary stats and shape for all weights in a layer.
 */
async function getLayerDetails(layer: Layer): Promise<Array<
    {name: string, stats: HistogramStats, shape: string, weight: tf.Tensor}>> {
  // TODO consider writing an async getWeights utility
  const weights = layer.getWeights();
  const layerVariables = layer.weights;
  const statsPromises = weights.map(tensorStats);
  const stats = await Promise.all(statsPromises);
  const shapes = weights.map(w => w.shape);
  return weights.map((weight, i) => ({
                       name: layerVariables[i].name,
                       stats: stats[i],
                       shape: formatShape(shapes[i]),
                       weight,
                     }));
}

function formatShape(shape: number[]): string {
  const oShape: Array<number|string> = shape.slice();
  if (oShape[0] === null) {
    oShape[0] = 'batch';
  }
  return `[${oShape.join(',')}]`;
}
