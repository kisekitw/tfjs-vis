import * as tf from '@tensorflow/tfjs';
import * as tfvis from '../../src'
import {getModel, loadData} from './model';

window.tf = tf;
window.tfvis = tfvis;

window.data;
window.model;

async function initData() {
  window.data = await loadData();
}

function initModel() {
  window.model = getModel();
}

function setupListeners() {
  document.querySelector('#load-data').addEventListener('click', async (e) => {
    await initData();
    document.querySelector('#show-examples').disabled = false;
    document.querySelector('#start-training-1').disabled = false;
    e.target.disabled = true;
  });
}

document.addEventListener('DOMContentLoaded', function() {
  initModel();
  setupListeners();
});
