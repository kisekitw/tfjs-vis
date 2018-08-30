import {History} from '@tensorflow/tfjs-layers/dist/base_callbacks';
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
export function history(
    container: Drawable, history: HistoryLike, metrics: string[]) {
  // Get the draw surface
  const drawArea = getDrawArea(container);

  // Format Data

  let values: Point2D[][];
  if (Array.isArray(history)) {
    values = metrics.map(metric => {
      return history.map((log, x) => ({x, y: log[metric]}));
    });
  } else {
    // Is a history object
    values = metrics.map(metric => {
      return (history.history[metric] as number[]).map((y, x) => ({x, y}));
    });
  }

  // Dispatch to render func
  const series = metrics;
  renderLinechart({values, series}, drawArea, {
    xLabel: 'Iteration',
    yLabel: 'Value',
  });
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

  for (const callbackName of callbackNames) {
    for (const metric of metrics) {
      const accumulatorName = getAccumulatorName(metric, callbackName);
      accumulators[accumulatorName] = [];
    }
  }

  function makeCallbackFor(callbackName: string) {
    return async (_: number, log: Logs) => {
      for (const metric of metrics) {
        const accumulatorName = getAccumulatorName(metric, callbackName);
        const metricLog = accumulators[accumulatorName];
        metricLog.push({[accumulatorName]: log[metric]});

        const subContainer = subSurface(drawArea, accumulatorName);
        history(subContainer, metricLog, [accumulatorName]);

        await nextFrame();
      }
    };
  }

  const callbacks = callbackNames.reduce((memo: FitCallbackHandlers, name) => {
    memo[name] = makeCallbackFor(name);
    return memo;
  }, {});

  return callbacks;
}

type HistoryLike = Logs[]|History;
interface FitCallbackHandlers {
  [key: string]: (iteration: number, log: Logs) => Promise<void>;
}

function getAccumulatorName(metric: string, callbackName: string): string {
  return `${metric}—${callbackName}`;
}
