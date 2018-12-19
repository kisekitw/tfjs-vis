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

import embed, {Mode, VisualizationSpec} from 'vega-embed';
import {ExtendedLayerSpec} from 'vega-lite/src/spec';

import {Drawable, HeatmapData, VisOptions,} from '../types';

import {getDrawArea} from './render_utils';

/**
 * Renders a heatmap.
 *
 * @param data Data consists of an object with a 'values' property
 *  and a 'labels' property.
 *  {
 *    // a matrix of numbers
 *    values: number[][],
 *
 *    // Human readable labels for each class in the matrix. Optional
 *    xLabels?: string[]
 *    yLabels?: string[]
 *  }
 *  e.g.
 *  {
 *    values: [[80, 23], [56, 94]],
 *    xLabels: ['dog', 'cat'],
 *    yLabels: ['size', 'temperature'],
 *  }
 * @param container An `HTMLElement` or `Surface` in which to draw the chart
 * @param opts optional parameters
 * @param opts.colorMap which colormap to use. One of viridis|blues|greyscale.
 *     Defaults to viridis
 * @param opts.domain a two element array representing a custom output domain
 *     for the color scale. Useful if you want to plot multiple heatmaps using
 *     the same scale.
 * @param opts.width width of chart in px
 * @param opts.height height of chart in px
 * @param opts.fontSize fontSize in pixels for text in the chart
 */
export async function renderHeatmap(
    data: HeatmapData, container: Drawable,
    opts: VisOptions&
    {colorMap?: ColorMap, domain?: number[]} = {}): Promise<void> {
  const options = Object.assign({}, defaultOpts, opts);
  const drawArea = getDrawArea(container);

  // Format data for vega spec; an array of objects, one for for each cell
  // in the matrix.
  const values: MatrixEntry[] = [];

  const inputArray = data.values;
  const {xLabels, yLabels} = data;

  for (let i = 0; i < inputArray.length; i++) {
    for (let j = 0; j < inputArray[i].length; j++) {
      const x = xLabels ? xLabels[i] : `${i}`;
      const y = yLabels ? yLabels[j] : `${j}`;
      const count = inputArray[i][j];
      values.push({
        x,
        y,
        count,
      });
    }
  }

  const embedOpts = {
    actions: false,
    mode: 'vega-lite' as Mode,
    defaultStyle: false,
  };

  const spec: VisualizationSpec = {
    'width': options.width || drawArea.clientWidth,
    'height': options.height || drawArea.clientHeight,
    'padding': 5,
    'autosize': {
      'type': 'fit',
      'contains': 'padding',
      'resize': true,
    },
    'config': {
      'axis': {
        'labelFontSize': options.fontSize,
        'titleFontSize': options.fontSize,
      },
      'text': {'fontSize': options.fontSize},
      'legend': {
        'labelFontSize': options.fontSize,
        'titleFontSize': options.fontSize,
      },
      'scale': {'bandPaddingInner': 0, 'bandPaddingOuter': 0},
    },
    'data': {'values': values},
    'encoding': {
      'x': {
        'field': 'x',
        'type': 'ordinal',
        // Maintain sort order of the axis if labels is passed in
        'scale': {'domain': xLabels},
        'title': options.xLabel,
      },
      'y': {
        'field': 'y',
        'type': 'ordinal',
        // Maintain sort order of the axis if labels is passed in
        'scale': {'domain': yLabels},
        'title': options.yLabel,
      },
    },
    'layer': [{
      'mark': {'type': 'rect'},
      'encoding': {
        'fill': {
          'field': 'count',
          'type': 'quantitative',
        },
      },
    }]
  };

  let colorRange: string[]|string;
  switch (options.colorMap) {
    case 'blues':
      colorRange = ['#f7fbff', '#4292c6'];
      break;
    case 'greyscale':
      colorRange = ['#000000', '#ffffff'];
      break;
    case 'viridis':
    default:
      colorRange = 'viridis';
      break;
  }
  if (colorRange !== 'viridis') {
    const fill = (spec.layer[0] as ExtendedLayerSpec).encoding!.fill!;
    // @ts-ignore
    fill.scale = {'range': colorRange};
  }

  if (options.domain) {
    const fill = (spec.layer[0] as ExtendedLayerSpec).encoding!.fill!;
    // @ts-ignore
    if (fill.scale != null) {
      // @ts-ignore
      fill.scale = Object.assign({}, fill.scale, {'domain': options.domain});
    } else {
      // @ts-ignore
      fill.scale = {'domain': options.domain};
    }
  }

  await embed(drawArea, spec, embedOpts);
  return Promise.resolve();
}

const defaultOpts = {
  xLabel: null,
  yLabel: null,
  xType: 'nominal',
  yType: 'nominal',
  colorMap: 'viridis',
  fontSize: 12,
  domain: null,
};

interface MatrixEntry {
  x: string;
  y: string;
  count: number;
}

type ColorMap = 'greyscale'|'viridis'|'blues';
