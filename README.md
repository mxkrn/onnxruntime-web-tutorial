# onnxruntime-web-tutorial

In this tutorial we will dive into `onnxruntime-web` by deploying a pre-trained *PyTorch* model to the browser. We will be using AlexNet as our deployment target. AlexNet has been trained as an image classifier on the <a href="https://www.image-net.org/">ImageNet dataset</a>, so I guess we will be building an image classifier. In part 1 of this tutorial we will focus on getting the inference engine working with some static image data. In part 2, we will integrate our inference engine into a *React* app and go into potential optimizations and pitfalls with using `onnxruntime-web`.