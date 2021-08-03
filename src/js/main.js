const ort = require("onnxruntime-web");

// ======================================================================
// Global variables
// ======================================================================

const WIDTH = 224;
const DIMS = [1, 3, WIDTH, WIDTH];
const MAX_LENGTH = DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3];
const MAX_SIGNED_VALUE = 255.0;
const classes = require("./imagenet_classes.json").data;

let predictedClass;
let isRunning = false;

// ======================================================================
// DOM Elements
// ======================================================================

const canvas = document.createElement("canvas"),
  ctx = canvas.getContext("2d");

document.getElementById("file-in").onchange = function (evt) {
  let target = evt.target || window.event.src,
    files = target.files;

  if (FileReader && files && files.length) {
    isRunning = true;
    var fileReader = new FileReader();
    fileReader.onload = () => onLoadImage(fileReader);
    fileReader.readAsDataURL(files[0]);
  }
};

const target = document.getElementById("target");
window.setInterval(function() {
  if (isRunning) {
    target.innerHTML = `<div class="align-self-start"><img src="src/images/loading.gif" class="loading"/>`;
  } else if (predictedClass !== "undefined") {
    target.innerHTML = `<h3>I think it's a ${predictedClass}!`;
  } else {
    target.innerHTML = ``;
  }
}, 500)

// ======================================================================
// Functions
// ======================================================================

function onLoadImage(fileReader) {
  var img = document.getElementById("input-image");
  img.onload = () => handleImage(img, WIDTH);
  img.src = fileReader.result;
}

function handleImage(img, targetWidth) {
  ctx.drawImage(img, 0, 0);
  const resizedImageData = processImage(img, targetWidth);
  const inputTensor = imageDataToTensor(resizedImageData, DIMS);
  run(inputTensor);
}

function processImage(img, width) {
  const canvas = document.createElement("canvas"),
    ctx = canvas.getContext("2d");

  canvas.width = width;
  canvas.height = canvas.width * (img.height / img.width);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  document.getElementById("canvas-image").src = canvas.toDataURL();
  return ctx.getImageData(0, 0, width, width).data;
}

function imageDataToTensor(data, dims) {
  // 1. filter out alpha
  // 2. transpose from [224, 224, 3] -> [3, 224, 224]
  const [R, G, B] = [[], [], []];
  for (let i = 0; i < data.length; i += 4) {
    R.push(data[i]);
    G.push(data[i + 1]);
    B.push(data[i + 2]);
    // here we skip data[i + 3] because it's the alpha channel
  }
  const transposedData = R.concat(G).concat(B);

  // convert to float32
  let i,
    l = transposedData.length; // length, we need this for the loop
  const float32Data = new Float32Array(MAX_LENGTH); // create the Float32Array for output
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / MAX_SIGNED_VALUE; // convert to float
  }

  // return ort.Tensor
  const inputTensor = new ort.Tensor("float32", float32Data, dims);
  return inputTensor;
}

function argMax(arr) {
  let max = arr[0];
  let maxIndex = 0;
  for (var i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }
  return [max, maxIndex];
}

async function run(inputTensor) {
  try {
    const session = await ort.InferenceSession.create("src/assets/model.onnx");
    const feeds = { input1: inputTensor };

    // feed inputs and run
    const results = await session.run(feeds);
    const [maxValue, maxIndex] = argMax(results.output1.data);

    predictedClass = `${classes[maxIndex]}`;
    isRunning = false;
  } catch (e) {
    console.error(e);
    isRunning = false;
  }
}
