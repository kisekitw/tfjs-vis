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

import {Logs} from '@tensorflow/tfjs-layers/dist/logs';

import {renderLinechart} from '../render/linechart';
import {getDrawArea, nextFrame} from '../render/render_utils';
import {Drawable, Point2D} from '../types';
import {subSurface} from '../util/dom';

/**
 * Renders a tf.Model training 'History'.
 *
 * @param container A `{name: string, tab?: string}` object specifying which
 *  surface to render to.
 * @param history A history like object. Either a tfjs-layers `History` object
 *    or an array of tfjs-layers `Logs` objects.
 * @param metrics An array of strings for each metric to plot from the history
 *    object. Using this allows you to control which metrics appear on the same
 *    plot.
 */
export async function history(
    container: Drawable, history: HistoryLike,
    metrics: string[]): Promise<void> {
  // Get the draw surface
  const drawArea = getDrawArea(container);

  // We organize the data from the history object into discrete plot data
  // objects so that we can group together appropriate metrics into single
  // multi-series charts.
  const plots: HistoryPlotData = {};
  for (const metric of metrics) {
    if (metric.match('loss')) {
      const values = getValues(history, metric);
      initPlot(plots, 'loss');
      plots['loss'].series.push(metric);
      plots['loss'].values.push(values);
    } else if (metric.match('acc')) {
      const values = getValues(history, metric);
      initPlot(plots, 'acc');
      plots['acc'].series.push(metric);
      plots['acc'].values.push(values);
    } else {
      const values = getValues(history, metric);
      initPlot(plots, metric);
      plots[metric].series.push(metric);
      plots[metric].values.push(values);
    }
  }

  // console.log('plots', plots);

  // Render each plot specified above to a new subsurface.
  // A plot may have multiple series.
  const plotNames = Object.keys(plots);
  const renderPromises = [];
  for (const name of plotNames) {
    const subContainer = subSurface(drawArea, name);
    const series = plots[name].series;
    const values = plots[name].values;
    const done = renderLinechart({values, series}, subContainer, {
      xLabel: 'Iteration',
      yLabel: 'Value',
    });
    renderPromises.push(done);
  }
  await Promise.all(renderPromises);
}

type HistoryLike = Logs[]|{
  history: {
    [key: string]: number[],
  }
};

interface HistoryPlotData {
  [name: string]: {
    series: string[],
    values: Point2D[][],
  };
}

function initPlot(plot: HistoryPlotData, name: string) {
  if (plot[name] == null) {
    plot[name] = {series: [], values: []};
  }
}

function getValues(history: HistoryLike, metric: string): Point2D[] {
  if (Array.isArray(history)) {
    console.log('getValues', metric, history);
    const points: Point2D[] = [];
    history.forEach((log: Logs, x: number) => {
      if (log[metric] != null) {
        const iteration = log._iteration != null ? log._iteration : x;
        points.push({x: iteration, y: log[metric]});
      }
    });
    return points;
  } else {
    return (history.history[metric] as number[]).map((y, x) => ({x, y}));
  }
}

/**
 * Returns a collection of callbacks to pass to tf.Model.fit. Callbacks are
 * returned for the following events, `onBatchEnd` & `onEpochEnd`.
 *
 * @param container A `{name: string, tab?: string}` object specifying which
 *  surface to render to.
 * @param metrics List of metrics to plot.
 */
export function fitCallbacks(
    container: Drawable, metrics: string[]): FitCallbackHandlers {
  const accumulators: {[key: string]: Logs[]} = {};
  const callbackNames = ['onEpochEnd', 'onBatchEnd'];
  const drawArea = getDrawArea(container);

  // Create an array to store logs for each callback.
  for (const callbackName of callbackNames) {
    const accumulatorName = getAccumulatorName(callbackName);
    accumulators[accumulatorName] = [];
  }

  function makeCallbackFor(callbackName: string) {
    return async (iteration: number, log: Logs) => {
      console.log(callbackName, iteration, log)
      // We want to store all the metrics for a given callback in the same array
      const accumulatorName = getAccumulatorName(callbackName);
      const metricLog = accumulators[accumulatorName];
      for (const metric of metrics) {
        metricLog.push({[metric]: log[metric], _iteration: iteration});
      }
      const subContainer =
          subSurface(drawArea, accumulatorName, {title: accumulatorName});
      history(subContainer, metricLog, metrics);
      await nextFrame();
    };
  }

  const callbacks: FitCallbackHandlers = {};
  callbackNames.forEach((name: string) => {
    callbacks[name] = makeCallbackFor(name);
  });
  return callbacks;
}

interface FitCallbackHandlers {
  [key: string]: (iteration: number, log: Logs) => Promise<void>;
}

function getAccumulatorName(callbackName: string): string {
  return `${callbackName}`;
}
