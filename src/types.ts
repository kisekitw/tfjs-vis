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

// Types shared across the project and that users will commonly interact with

/**
 * The public visor api
 */
export interface Visor {
  el: HTMLElement;
  surface: (options: SurfaceInfo) => Surface;
  isFullscreen: () => boolean;
  isOpen: () => boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  toggleFullScreen: () => void;
  unbindKeys: () => void;
  bindKeys: () => void;
}

/**
 * The public api of a 'surface'
 */
export interface Surface {
  container: HTMLElement;
  label: HTMLElement;
  drawArea: HTMLElement;
}

/**
 * Options used to specify a surface.
 *
 * name and tab are also used for retrieval of a surface instance.
 */
export interface SurfaceInfo {
  name: string;
  tab?: string;
  styles?: StyleOptions;
}

/**
 * Internally all surfaces must have a tab.
 */
export interface SurfaceInfoStrict extends SurfaceInfo {
  name: string;
  tab: string;
  styles?: StyleOptions;
}

/**
 * Style properties are generally optional as components will specify defaults.
 */
export type StyleOptions = Partial<CSSOptions>;
export interface CSSOptions {
  width: string;
  height: string;
  maxWidth: string;
  maxHeight: string;
}
