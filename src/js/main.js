const ort = require("onnxruntime-web");

// ======================================================================
// Global variables
// ======================================================================

const WIDTH = 224;
const DIMS = [1, 3, WIDTH, WIDTH];
const MAX_LENGTH = DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3];

const classes = require("./imagenet_classes.json").data;
let imageData;

// ======================================================================
// DOM Elements
// ======================================================================

const canvas = document.createElement('canvas'),
    ctx = canvas.getContext("2d"),
    oc = document.createElement('canvas'),
    octx = canvas.getContext('2d');

document.getElementById("file-in").onchange = function (evt) {
  let target = evt.target || window.event.src,
    files = target.files;

  // FileReader support
  if (FileReader && files && files.length) {
      var fileReader = new FileReader();
      fileReader.onload = () => showImage(fileReader);
      fileReader.readAsDataURL(files[0]);
  }
}

const target = document.getElementById("target");
const altTargets = [];
for (let i = 0; i < 4; i++) {
  altTargets.push(document.getElementById(`alt-target${i}`));
}

// ======================================================================
// Functions
// ======================================================================

function showImage(fileReader) {
    var img = document.getElementById("input-image");
    img.onload = () => handleImageData(img, WIDTH);
    img.src = fileReader.result;
}

function handleImageData(img, targetWidth) {
  // draw original image
  ctx.drawImage(img, 0, 0);

  // resize and draw resized image
  imageData = resizeImage(img, targetWidth)

  // use resized image data to constructor tensor
  const inputTensor = imageDataToTensor(imageData, DIMS);

  // run
  run(inputTensor, DIMS);
}

function resizeImage(img, width) {
  const canvas = document.createElement('canvas'),
    ctx = canvas.getContext("2d")

  canvas.width = width;
  canvas.height = canvas.width * (img.height / img.width);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  document.getElementById("canvas-image").src = canvas.toDataURL();
  return ctx.getImageData(0, 0, width, width).data;
}

function imageDataToTensor(data, dims) {
  // filter out alpha channels
  const filteredData = data.filter((v, i) => i % 4 != 3)
  if (MAX_LENGTH != filteredData.length) {
    console.error(`Mismatching data sizes: ${MAX_LENGTH} != ${filteredData.length}`)
  }

  // transpose from [224, 224, 3] -> [3, 224, 224]
  const [R, G, B] = [[], [], []]
  for (let i = 0; i < imageData.length; i += 4) {
    R.push(imageData[i]);
    G.push(imageData[i + 1]);
    B.push(imageData[i + 2]);
  }

  const processedData = R.concat(G).concat(B);

  // convert to float32
  let i, l = processedData.length; // length, we need this for the loop
  const tensorData = new Float32Array(MAX_LENGTH); // create the Float32Array for output
  for (i = 0; i < l; i++) {
    tensorData[i] = processedData[i] / 255.0; // convert to float
  }

  // create tensor
  const inputTensor = new ort.Tensor('float32', tensorData, dims);
  return inputTensor;
}

function indexOfMax(arr) {
  let sorted = [];
  let max = arr[0];
  let maxIndex = 0;

  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
          sorted.push(maxIndex);
      }
  }

  return [maxIndex, sorted.reverse()];
}

async function run(inputTensor, dims) {
  try {
    // create a new session and load the AlexNet model.
    const session = await ort.InferenceSession.create('src/assets/model.onnx');

    // prepare feeds. use model input names as keys.
    const feeds = { input1: inputTensor };

    // feed inputs and run
    const results = await session.run(feeds);
    const [maxIdx, sortedMaxIdx] = indexOfMax(results.output1.data);
    target.innerHTML = classes[maxIdx];
  } catch (e) {
    console.error(e);
  }
}