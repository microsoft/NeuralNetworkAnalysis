
Neural Network Analysis Framework
=================================

The repository contains an abstract interpreter for deep neural networks that has been used for generation of adversarial examples that satisfy various user-specified conditions. The framework supports many types of common CNN layers, and for the moment can process protobuf-based Caffe models. The description of constraint solving to generate adversarial examples can be found in the paper:

[Measuring Neural Net Robustness with Constraints](https://arxiv.org/abs/1605.07262)

by Osbert Bastani, Yani Ioannou, Aditya Nori, Leonidas Lampropoulos, Dimitrios Vytiniotis, and Antonio Criminisi. The paper has appeared in NIPS 2016. 
The results in the NIPS paper have been produced using this framework.

What's here
================================

* `protobuf/` Simple utility that converts protobuf-based .caffemodel files to our own binary format.
* `src/` The actual C# codebase.


To build the protobuf utility
-----------------------------
Just follow the README file in the `protobuf/` directory


To build the analysis framework (Windows, VS 2015)
------------------------------------------------------

1) You need to have Gurobi (an industrial linear solver) and its .NET bindings. Microsoft Solver Foundation does come with an old version of Gurobi with .NET bindings and this is the one that we 
have used for experiments, so you can simply install Microsoft Solver Foundation, available [here](https://msdn.microsoft.com/en-us/devlabs/hh145003). Alternatively, it is extremely easy to change the solver but you need to do some actual hacking. 

2) Open the solution `src/NNAnalysis/NNAnalysis.sln` 

    You will have to install some NuGet packages and add references to the MSF and Gurobi assemblies, wherever they have been installed in your system. The extra NuGet packages you 
    need are:

    1. `Mono.Options` (Command line argument parsing) 
    2. `MathNet.Numerics` (Linear algebra, including MKL and OpenBlas)

    Install all of them to all projects in the solution. You can configure programmatically if you want MKL or OpenBlas. At the moment default active is MKL.

3) Build the solution. The solution contains one library `NNAnalysis` with pretty much all of the functionality, and several executable projects. The ones that are most important for counterexample generation are `NNMNISTAccuracy` (targetting MNIST) and `NNCIFARAccuracy` (targetting CIFAR-10).

It should be possible to build under Mono in Linux or Mac but we have not tried.

How to *actually* run stuff? 
============================

Let's work with the example of CIFAR-10.

Step 1: Make sure you have the dataset and a trained Caffe model
-----------------------------------------------------------------

For CIFAR-10 there's one data set fo testing (`test_batch.bin`) and 5 data batches
for training. In a caffe installation the are somewhere like `caffe/data/cifar10`.
Then follow standard Caffe instructions and train your model to produce a `.caffemodel` 
file. NB: Make sure that your Caffe model encoding is with protocol buffers. 

Step 2: Convert your model to a format we like
-----------------------------------------------
Run the utility in `protobuf/` according to the README there on your caffe model
file to produce a binary file that our tool can process. Let's call the new file 
`cifar10.mldiffmodel`. 

Our tool can process a lot of standard CNN layers but don't try anything too crazy
with the caffe model (like a cross-layer residual architecture). We've only implemented 
straight layer-by-layer functionality.

In the future we'd like to directly accept protocol buffer descriptions of caffe models,
but for the moment it's an extra step you have to perform.

Step 3: Find adversarial examples
----------------------------------

You are good to try and find adversarial examples.

1) Go to the directory where `NNCIFARAccuracy.exe` lives. For instance
   if you compile a Release executable for AnyCPU, that would be something like:
   ```
       NNAnalysis\NNCIFARAccuracy\bin\Release\NNCIFARAccuracy.exe
   ```
2) Once in that directory just type:
   ```
     .\NNCIFARAccuracy.exe --help
   ```
   You will see a bunch of options. The most basic configuration would be:
   ```
    .\NNCIFARAccuracy.exe --nnet cifar10.mldiffmodel --dataset .\test_batch.bin
   ``` 
   where `cifar10.mldiffmodel` is your model (See Step 2), and the dataset is the actual CIFAR-10 test batch or even one of the training batches if you wish. You can use whatever you like.

    If the tool seems to be doing something, without throwing weird exceptions, congratulations, 
    go to Step 4 because you already started producing adversarial examples!

There are a whole lot of option on our tools -- for example if you _only_ want to evaluate the accuracy on this data batch you can say:
```
.\NNCIFARAccuracy.exe --nnet cifar10.mldiffmodel --dataset .\test_batch.bin --only-accuracy
```
Hopefully you will get a number very close or exactly the same to what
Caffe gave you when you tested the network. If not, there's most
likely a bug in one of our layers or check in your Caffe model prototxt 
that you are not using the following (currently unsupported) features:
1. Convolutions with stride != 1 
2. Convolutions with the group parameter set and > 1 
3. LRN layers

Our tool chain does not support those.

Step 4: Understand what you run
--------------------------------

If you have reached so far, you managed to build and run the tool, so
let's see what it produces. It iterates through the dataset and for each data instance  and label it tries to generate a counterexample. By counterexample, we mean an instance that has the 2nd best label, as we can't express disjunctive constraints: 2nd best OR 3rd best OR 4th best etc.
There are various objectives exposed in the options that one can experiment with regards to the nature of the counterexemple, e.g. the bounding ball (and the distance measure), minimization or just model-finding, and of course other linear properties can actually be encoded with some actual
hacking.

As the tool runs, you will see a long and amusing trace explaining what is happening: i.e. symbolically interpreting the network (i.e. doing a forward pass) and generatind constraints, solving them, iterating if we find a spurious counterexample etc.

Occassionally you will see a "Real counterexample" message. That means we did manage to find a real counterexample. Yay!

Counterexamples are logged in a new local CSV file called by default:
```
generic-registry.csv
```
Furthemore, an additional directory:
```
generic-registry/
```
contains actual PNG images (both the original, and the synthesized one, and with an option their difference pixel by pixel as a png image).

If you wish to use a different name for the registry CSV file and the corresponding
directory, just use a different `--registry` value on the command line, say:
```
--registry=my-cifar10-first-registry
```
Then our tool will be populating `my-cifar10-first-registry.csv` and dumping png images in the `my-cifar10-first-registry/` directory. Each line in the counterexample registry is a comma-separated list of the following things:
1. Name of this registry,
2. Original training point .png filename,
3. Original label,
4. Synthesized counterexample .png filename,
5. New label,
6. L-infinity distance between orig. and synth.,
7. L1 distance between orig. and synth.,
8. Confidence of the original classification,
9. Difference between the most confident 
10.  classification (above number) and the second best for the orig. image,
11. Confidence of the synthesized image,
12. Difference between the most confident classification (above number) and the second best for the synthesized image.
     
If you let the tool run for a while you will see pngs populating the
registry directory and you will see more and more entries appearing in
the .csv file.

Fore more options check the help `--help` and browse the code! 

Contributing
============

Please read [here](CONTRIBUTING.md)
