# onnxruntime-web-tutorial

![alexnet-image-classifier](https://user-images.githubusercontent.com/17745834/128233563-8bb289c8-a6e7-48b9-83cc-c4ad54ad11f3.png)

#### 📄 **Tutorial** - https://rekoil.io/blog/onnxruntime-web-tutorial/

This repository contains the code for the tutorial on building a browser-native machine learning app using ONNX Runtime Web. In the tutorial, we dive into `onnxruntime-web` by deploying a pre-trained *PyTorch* model. The model is run natively in the browser using WebAssembly via `onnxruntime-web`. The model that in use is AlexNet which, has been trained as an image classifier on the <a href="https://www.image-net.org/">ImageNet dataset</a>.

To launch a live server locally, simply use `npm run serve`. Note that `npm run serve` also called `npm run build` meaning that any changes in `main.js` or other package dependencies will be included in the deployed app bundle.

The app uses `webpack` to bundle the package dependencies into `dist/bundle.min.js`, to re-build the bundle: `npm run build`.
