# Building a browser-native machine learning app using ONNX Runtime Web

Deploying machine learning models outside of a _Python_ environment used to be difficult. When the target platform is the browser, the defacto standard for serving predictions has been an API call to a server-side inference engine. For many different reasons, server-side inference engines are slowly moving out of the vogue and machine learning models are more often being deployed natively. `Tensorflow` has done a good job at supporting this movement by providing cross-platform APIs, however none of us want to be married to a single ecosystem.

In comes the <a href="https://www.onnxruntime.ai/">Open Neural Network Exchange</a> (ONNX) project, driven by _Microsoft_, which has been seeing massive development efforts and is slowly reaching a stable state. It's now easier than ever to deploy machine-learning models, trained using your machine-learning framework of choice, on a number of platforms including _C, C++, Javascript, Java_, and several others, with hardware acceleration out of the box. In April this year, `onnxruntime-web` was introduced (see this <a href="https://github.com/microsoft/onnxruntime/pull/7394">Pull Request</a>). `onnxruntime-web` uses _WebAssembly_ to compile the `onnxruntime` inference engine to run _ONNX_ models in the browser. It's about _WebAssembly_ time starts to flex its muscles, especially when paired with _WebGL_ we suddenly have GPU-powered machine learning in the browser, pretty cool.

In this tutorial we will dive into `onnxruntime-web` by deploying a pre-trained _PyTorch_ model to the browser. We will be using AlexNet as our deployment target. AlexNet has been trained as an image classifier on the <a href="https://www.image-net.org/">ImageNet dataset</a>, so we will be building an image classifier - nothing better than re-inventing the wheel. In part 1 of this tutorial, we will focus on the browser by deploying a simple static site allowing users to classify their images using _AlexNet_.

<!-- In part 2, we will see how we can integrate our inference engine into real-life Javascript apps based on frameworks like *React*, *React Native*, and *Node.js*. -->

## Prerequisite

You will need a trained machine-learning model exported as an _ONNX_ binary protobuf file. There's many ways to achieve this using a number of different deep-learning frameworks. For the sake of this tutorial, I will be using the exported model from the _AlexNet_ example in the <a href="https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx">PyTorch documentation</a>, the python code snippet below will help you generate your own model. You can also follow the linked documentation to export your own _PyTorch_ model. If you're coming from _Tensorflow_, <a href="https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/tensorflow-convert-model">this tutorial</a> will help you with exporting your model to _ONNX_. Lastly, _ONNX_ doesn't just pride itself on cross-platform deployment, but also in allowing exports for a multitude of deep-learning frameworks, so those of you using another exotic framework should be able to find support for exporting to _ONNX_ in the relevant docs.

```python:onnx_model.py
import torch
import torchvision

dummy_input = torch.randn(1, 3, 224, 224)
model = torchvision.models.alexnet(pretrained=True)

input_names = ["input1"]
output_names = ["output1"]

torch.onnx.export(
  model,
  dummy_input,
  "alexnet.onnx",
  verbose=True,
  input_names=input_names,
  output_names=output_names
)
```

Running this file creates a file, `alexnet.onnx`, a binary protobuf file which contains both the network structure and parameters of the model you exported (in this case, AlexNet).

## ONNX Runtime Web

> _ONNX Runtime Web_ is a Javacript library for running _ONNX_ models on the browser and on _Node.js_. ONNX Runtime Web has adopted _WebAssembly_ and _WebGL_ technologies for providing an optimized ONNX model inference runtime for both CPUs and GPUs.

Sounds like our cup of tea.

The official package is hosted on _npm_ under the name `onnxruntime-web`. When using a bundler or working server-side, this package can be installed using `npm install`. However, it's also possible to deliver the code via a CDN using a script tag. The bundling process is a bit more involved so we will start with the script tag approach and come back to using the _npm_ package later.

### Runtime

Let's start with the foundation - `onnxruntime` exposes a runtime object called an `InferenceSession` with a method `.run()` which is used to initiate the forward pass with the desired inputs. Both the `InferenceSessesion` constructor and the accompanying `.run()` method return a `Promise` so we will run the entire process inside an `async` context. Before implementing any browser elements, we will check that our model runs with a dummy input tensor, remembering the input and output names and sizes that we defined earlier when exporting the model. The code below should be saved in a file called `main.js`.

```js:main.js
async function run() {
  try {
    // create a new session and load the AlexNet model.
    const session = await ort.InferenceSession.create('./alexnet.onnx');

    // prepare dummy input data
    const dims = [1, 3, 224, 224];
    const size = dims[0] * dims[1] * dims[2] * dims[3];
    const inputData = Float32Array.from({ length: size }, () => Math.random());

    // prepare feeds. use model input names as keys.
    const feeds = { input1: new ort.Tensor('float32', inputData, dims) };

    // feed inputs and run
    const results = await session.run(feeds);
    console.log(results.output1.data);
  } catch (e) {
    console.log(e);
  }
}
run();
```

We then implement a simple HTML template, `index.html`, in the same directory as `alexnet.onnx` and `main.js`.

```html:index.html
<!DOCTYPE html>
<html>
  <header>
    <title>ONNX Runtime Web Tutorial</title>
  </header>
  <body>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js">
    </script>
    <script src="main.js"></script>
  </body>
</html>
```

Finally, to run this we can use `live-server`. If you haven't started an `npm` project by now, please do so by running `npm init` in your current working directory. Once you've completed the setup, install live-server (`npm install live-server`) and serve the static HTML page using `npx light-server -s . -p 8080`.

Congratulations! You're now running a machine learning model natively in the browser. To check that everything is running fine simply go to your console and make sure that the output tensor is logged (AlexNet is pretty big so it's normal that inference takes a few seconds).

### Bundled deployment

Next we will use `webpack` to bundle our dependencies as would be the case if we want to deploy the model in a _Javascript_ app powered by frameworks like _React_ or _Vue_. Usually bundling is a relatively simple procedure, however `onnxruntime-web` requires a slightly more involved `webpack` configuration because _WebAssembly_ is used to provide the natively assembled runtime. The following steps are based on the examples provided by the official <a href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-bundler">_ONNX_ documentation</a>.

Assuming you've already started an _npm_ project (using `npm init`), we first install the dependencies.

1. `npm install onnxruntime-web`
2. `npm install -D webpack webpack-cli copy-webpack`

We then need to update `main.js` to use the new package instead of loading the `onnxruntime-web` module via a CDN. This is done by updating `main.js` with a one-liner at the start of the script.

`const ort = require('onnxruntime-web');`

We then save the below configuration file as `webpack.config.js` and run `npx webpack`.

```js:webpack.config.js
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = () => {
    return {
        target: ['web'],
        entry: path.resolve(__dirname, 'main.js'),
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: 'bundle.min.js',
            library: {
                type: 'umd'
            }
        },
        plugins: [new CopyPlugin({
            // Use copy plugin to copy *.wasm to output folder.
            patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
        })],
        mode: 'production'
    }
};
```

Finally, before reloading the live server, we will update `index.html`:

1. Remove the script tag to load `ort.min.js` from the CDN
2. Load the code dependencies from `bundle.min.js` (which contains all our dependencies bundled and minified) instead of `main.js`

`index.html` should now look something like this.

```
<!DOCTYPE html>
<html>
  <header>
    <title>ONNX Runtime Web Tutorial</title>
  </header>
  <body>
    <script src="bundle.min.js.js"></script>
  </body>
</html>
```

## Image Classifier

Let's put this model to work and setup our image classifier.

To start, will need some utility functions to load, resize, and display the image. We will use the `canvas` object for this. Additionally, image classification systems typically have lots of magic built into the pre-processing pipelines, this is quite trivial to implement in _Python_ using frameworks like `numpy`, unfortunately this is not the case _Javascript_. It follows that we will have to implement our pre-processing from scratch to transform our image data into the correct tensor format.

### 1. DOM Elements

To start, let's create the necessary HTML elements.

1. File input

```html
<label for="fileIn"><h2>What am I?</h2></label>
<input type="file" id="file-in" name="file-in" />
```

2. Image displays (we will display both the original and rescaled image)

```html
<img id="input-image" class="input-image"></img>
<img id="scaled-image" class="scaled-image"></img>
```

3. Display classification

```html
<h3 id="target"></h3>
```

### 2. Image load and display

We want to load an image from file and display it - moving to `main.js`, we will get the file input element and use `FileReader` to read the data into memory. Then, the image data will be passed to `handleImage` which will draw the image using the `canvas` context.

```js
const canvas = document.createElement("canvas"),
  ctx = canvas.getContext("2d");

document.getElementById("file-in").onchange = function (evt) {
  let target = evt.target || window.event.src,
    files = target.files;

  if (FileReader && files && files.length) {
    var fileReader = new FileReader();
    fileReader.onload = () => onLoadImage(fileReader);
    fileReader.readAsDataURL(files[0]);
  }
};

function onLoadImage(fileReader) {
  var img = document.getElementById("input-image");
  img.onload = () => handleImage(img);
  img.src = fileReader.result;
}

function handleImage(img) {
  ctx.drawImage(img, 0, 0);
}
```

### 2. Preprocess and convert image to tensor

Now that we can load and display an image, we want to move to extracting and processing the data. Remember that our model takes in a matrix of shape `[1, 3, 224, 224]`, this means we will probably have to resize the image and perhaps also transpose the dimensions depending on how we extract the image data.

To resize and extract image data, we will use the `canvas` context again. Let's define a function `processImage` that does this. `processImage` has the necessary elements in scope to immediately draw the scaled image so we will also do that here.

```js
function processImage(img, width) {
  const canvas = document.createElement("canvas"),
    ctx = canvas.getContext("2d");

  // resize image
  canvas.width = width;
  canvas.height = canvas.width * (img.height / img.width);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  document.getElementById("scaled-image").src = canvas.toDataURL();
  return ctx.getImageData(0, 0, width, width).data;
}
```

We can now add a line to the function `handleImage` which calls `processImage`.

```js
const resizedImageData = processImage(img, targetWidth);
```

Finally, let's implement a function called `imageDataToTensor` which applies the transforms needed to get the image data ready to be used as input to the model. `imageDataToTensor` should apply three transforms:

1. Filter out the alpha channel, our input tensor only contains 3 channels corresponding to RGB.
2. `ctx.getImageData` returns data in the shape `[224, 224, 3]` so we need to transpose the data to the shape `[3, 224, 224]`
3. `ctx.getImageData` returns a `UInt8ClampedArray` with `int` values ranging 0 to 255, we need to convert the values to `float32` and store them in a `Float32Array` to construct our tensor input.

```js
function imageDataToTensor(data, dims) {
  // 2. transpose from [224, 224, 3] -> [3, 224, 224]
  const [R, G, B] = [[], [], []];
  for (let i = 0; i < data.length; i += 4) {
    R.push(data[i]);
    G.push(data[i + 1]);
    B.push(data[i + 2]);
    // here we skip data[i + 3] corresponding to the alpha channel
  }
  const transposedData = R.concat(G).concat(B);

  // convert to float32
  let i,
    l = transposedData.length; // length, we need this for the loop
  const float32Data = new Float32Array(3 * 224 * 224); // create the Float32Array for output
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }

  const inputTensor = new ort.Tensor("float32", float32Data, dims);
  return inputTensor;
}
```

### 3. Display classification result

There's two last steps before we're ready to make some predictions and display our classification results.

1. Stitch together our image processing pipeline in `handleImageData`

```js
function handleImage(img, targetWidth) {
  ctx.drawImage(img, 0, 0);
  const resizedImageData = processImage(img, targetWidth);
  const inputTensor = imageDataToTensor(resizedImageData, DIMS);
  run(inputTensor);
}
```

2. Get the classification result by first getting the index of the maximum value in the output data, this is done using the `argMax` function.

```js
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
```

3. Update `run` to accept a tensor input. Secondly, we need to use the max index to actually retrieve the results from a list of <a href="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt">ImageNet classes</a>. I've pre-converted this list to JSON and we will load it into our script using `require`, you can find the JSON file in the code repository linked at the bottom.

```js
const classes = require("./imagenet_classes.json").data;

async function run(inputTensor) {
  try {
    const session = await ort.InferenceSession.create("./alexnet.onnx");

    const feeds = { input1: inputTensor };
    const results = await session.run(feeds);

    const [maxValue, maxIndex] = argMax(results.output1.data);
    target.innerHTML = `${classes[maxIndex]}`;
  } catch (e) {
    console.error(e); // non-fatal error handling
  }
}
```

## Conclusion

### Styling (Bonus)

Plain HTML gives me the shivers, so let's procrastinate a little and add some styling and layout formatting using Bootstrap 5. First, let's install bootstrap and re-compile the package bundle. We can now use bootstrap classes to put everything in it's place.

`npm install bootstrap && npx webpack`

Secondly, let's create a file called `main.css` in the working directory and add the following lines to the header of `index.html`.

```
<link rel="stylesheet" type="text/css" href="./main.css">
```

And from there on it's just jazz.

You can re-visit the full code repository and styling in the <a href="https://github.com/mxkrn/onnxruntime-web-tutorial">project repository</a>. If you want to take your deployment to the next level, have a look at <a href="#">Part 2</a> where we will go into deploying your model to _Node.js_, _React_, or _React Native_.

Thank you for reading!
