# Part 1: Building a browser-based machine learning apps using ONNX Runtime Web

Deploying machine learning models outside of a *Python* runtime used to be difficult, especially when the target platform is the browser the defacto standard for serving predictions has been an API call to a server-side inference engine. For many reasons we will not go into at this point in time, server-side inference engines are slowly moving out of the vogue and machine learning models are more often being deployed natively. `Tensorflow` has done a good job at supporting this movement by providing cross-platform APIs, unfortunately all the rest of us are left in the dark. 

In comes the <a href="https://www.onnxruntime.ai/">Open Neural Network Exchange</a> (ONNX) project, driven by *Microsoft*, which has been seeing massive development efforts and is slowly reaching a stable state. It's now easier than ever to deploy machine-learning models, trained using your machine-learning framework of choice, on a number of platforms including *C, C++, Javascript, Java*, and several others, with both CPU and GPU support out of the box. In April this year, `onnxruntime-web` was introduced (see this <a href="https://github.com/microsoft/onnxruntime/pull/7394">Pull Request</a>). `onnxruntime-web` uses *WebAssembly* to compile the `onnxruntime` inference engine to run *ONNX* models in the browser. It's about *WebAssembly* time starts to flex it's muscles, especially when paired with *WebGL* we suddenly have GPU-powered machine learning in the browser, pretty cool.

In this tutorial we will dive into `onnxruntime-web` by deploying a pre-trained *PyTorch* model to the browser. We will be using AlexNet as our deployment target. AlexNet has been trained as an image classifier on the <a href="https://www.image-net.org/">ImageNet dataset</a>, so I guess we will be building an image classifier. In part 1 of this tutorial we will focus on getting the inference engine working with some static image data. In part 2, we will integrate our inference engine into a *React* app and go into potential optimizations and pitfalls with using `onnxruntime-web`.

## Prerequisite

You will need a trained machine-learning model exported as an *ONNX* binary protobuf file. There's many ways to achieve this using a number of different deep-learning frameworks. For the sake of this tutorial, I will be using the exported model from the *AlexNet* example in the <a href="https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx">PyTorch documentation</a>, the python code snippet will help you generate your own model. You can also follow the linked documentation to export your own *PyTorch* model. If you're coming from *Tensorflow*, <a href="https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/tensorflow-convert-model">this tutorial</a> will help you with exporting your own model. Those of you using another frameworks should get by with a simple google search as *ONNX* doesn't just pride itself on cross-platform deployment, but also in allowing exports for a multitude of deep-learning frameworks.

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

The resulting `alexnet.onnx` is a binary protobuf file which contains both the network structure and parameters of the model you exported (in this case, AlexNet).

## ONNX Runtime Web

> *ONNX Runtime Web* is a Javacript library for running *ONNX* models on the browser and on *Node.js*. ONNX Runtime Web has adopted *WebAssembly* and *WebGL* technologies for providing an optimized ONNX model inference runtime for both CPUs and GPUs.

Sounds like our cup of tea. 

The official package is hosted on *npm* under the name `onnxruntime-web`. When using a bundler or working server-side, this package can be installed using `npm install`. However, it's also possible to deliver the code via a CDN using a script tag. The bundling process is a bit more involved so we will start with the script tag approach and come back to using the *npm* package later.

### Runtime

Let's start with the foundation - `onnxruntime` exposes a core runtime object called an `InferenceSession` with a method `.run()` which is used to initiate the forward pass with the desired inputs. Both the `InferenceSessesion` constructor and the accompanying `.run()` method return a `Promise` so we will run the entire process inside an `async` context. Before implementing any browser elements, we will check that our model runs with a dummy input tensor, remembering the input and output names that we defined earlier when exporting the model. The code below should be saved in a file called `main.js`.

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

Congratulations! You're now running our machine learning model in the browser using our local hardware. To check that everything is running fine simply go to your console and make sure that the output tensor is logged (AlexNet is pretty big so it's normal that it takes a few seconds).

### Bundled deployment

Next we will use `webpack` to bundle our dependencies as would be the case if we want to deploy the model in a *Javascript* framework like *React* or *Vue*. Usually bundling is a relatively simple procedure, however `onnxruntime-web` requires a slightly more involved `webpack` configuration because *WebAssembly* is used to provide the natively assembled runtime. The following steps are based on the examples provided by the official <a href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/quick-start_onnxruntime-web-bundler">*ONNX* documentation</a>.

Assuming you've already started an *npm* project (using `npm init`), we first install the dependencies.

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

Finally, before reloading the live server, we need to update `index.html` with two final steps:
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

Let's get to business and implement the components needed to load any image and spit out our classification. We just need three additional utility components.

1. Load and display image
2. Preprocess and convert image to tensor
3. Display our classification result

#### 1. Load and display an image

The user should be able to load an image from file, the input image should be loaded into memory and be displayed once the image has been uploaded.

#### 2. Preprocess and convert image to tensor

#### 3. Display classification result


---
### Styling (Bonus)

Plain HTML gives me the shivers, so let's procrastinate a little and add some styling and layout formatting using Bootstrap 5. First, let's install bootstrap and re-compile the package bundle. We can now use bootstrap classes to put everything in it's place.

`npm install bootstrap && npx webpack`

Secondly, let's create a file called `main.css` in the working directory and add the following lines to the header of `index.html`.

```
<link rel="stylesheet" type="text/css" href="./main.css">
```

And from there on it's just jazz.

You can re-visit the full code repository and styling in the <a href="https://github.com/mxkrn/onnxruntime-web-tutorial">project repository</a>. If you want to take your deployment to the next level, have a look at <a href="#">Part 2</a> where we will go into deploying your model to *Node.js*, *React*, or *React Native*.

Thank you for reading!