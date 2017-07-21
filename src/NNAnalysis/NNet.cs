// --------------------------------------------------------------------------------------------------
// Neural Network Analysis Framework
//
// Copyright(c) Microsoft Corporation
// All rights reserved.
//
// MIT License
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
//  associated documentation files (the "Software"), to deal in the Software without restriction,
//  including without limitation the rights to use, copy, modify, merge, publish, distribute,
//  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all copies or
//  substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// --------------------------------------------------------------------------------------------------


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

using System.Diagnostics;

namespace NNAnalysis
{

    public enum LayerType
    {
        RECTIFIED_LINEAR, INNER_PRODUCT, LOSS, CONVOLUTION_LAYER, POOLING_LAYER, VIRTUAL_LAYER, DATA_LAYER
    }

    /// <summary>
    ///  A layer of the neural network and the operations it supports
    /// </summary>
    public abstract class Layer
    {
        LayerType layerType_;
        int inputDimension_;
        int outputDimension_;
        int index_; // Which layer are we?

        ImageCoordinates inputCoordinates_;  // NB: null means there is no particular structure in the input
        ImageCoordinates outputCoordinates_; // NB: null means there is no particular structure in the output

        public void InitLayer(
            int index, 
            LayerType layerType, 
            int inputDimension, 
            int outputDimension, 
            ImageCoordinates inputCoordinates, 
            ImageCoordinates outputCoordinates)
        {
            index_ = index;
            layerType_ = layerType;
            inputDimension_ = inputDimension;
            outputDimension_ = outputDimension;
            inputCoordinates_ = inputCoordinates;
            outputCoordinates_ = outputCoordinates;
        }


        public int Index
        {
            get { return index_; }
        }
        public LayerType LayerType
        {
            get { return layerType_; }
        }
        public int InputDimension
        {
            get { return inputDimension_; }
        }
        public int OutputDimension
        {
            get { return outputDimension_; }
        }
        public ImageCoordinates InputCoordinates
        {
            get { return inputCoordinates_; }
        }
        public ImageCoordinates OutputCoordinates
        {
            get { return outputCoordinates_; }
        }
        public abstract LPSTerm[] EvaluateSymbolic(LPSState state, LPSTerm[] input);
        public abstract Vector<double> EvaluateConcrete(Vector<double> input);
        public abstract void Instrument(NNInstrumentation instrumentation, Vector<double> input, Vector<double> output);

        public abstract bool IsAffine();

    }

    /// <summary>
    /// A neural network: just a collection of layers
    /// </summary>
    public class NeuralNet
    {
        // We elevate cropping to a first-class citizen of a neural network
        // to expose it to the symbolic evaluator. If cropT == null, then no 
        // cropping happens.
        CropTransform cropT = null; 

        List<Layer> layers_ = new List<Layer>();

        public void AddCropTransform(CropTransform crop)
        {
            cropT = crop;
        }

        public void AddLayer(Layer layer)
        {
            layers_.Add(layer);
        }
        public List<Layer> Layers
        {
            get { return layers_; }
        }
        public int LayerCount
        {
            get { return layers_.Count; }
        }
        public int InputDimensionPostCrop
        {
            get { return layers_[0].InputDimension; }
        }

        public int InputDimensionPreCrop
        {
            get 
            { 
                if (cropT != null) return cropT.OriginalDimension();
                return layers_[0].InputDimension;
            }
        }

        public Vector<double> CropMaybe(Vector<double> image)
        {
            if (cropT != null) { return cropT.Transform(image); }
            return image;
        }

        public Vector<double> UnCropMaybe(Vector<double> orig, Vector<double> image)
        {
            if (cropT != null) { return cropT.UnTransform(orig, image); }
            return image;
        }

        public List<LayerType> LayerTypes()
        {
            List<LayerType> layerTypes = new List<LayerType>();
            for (int i = 0; i < LayerCount; i++)
            {
                layerTypes.Add(Layers[i].LayerType);
            }
            return layerTypes;
        }
        public double[] EvaluateNNConcretePostCrop(double[] input, NNInstrumentation instr = null)
        {
            return EvaluateNNConcretePostCrop(DenseVector.OfArray(input), instr);
        }
        public double[] EvaluateNNConcretePostCrop(Vector<double> input, NNInstrumentation instr)
        {
            Vector<double> v = input;
            for (int i = 0; i < LayerCount; i++)
            {
                Layer curr = Layers[i];
                var w = curr.EvaluateConcrete(v);
                if (instr != null) curr.Instrument(instr, v, w);
                v = w;
            }
            return v.ToArray();

        }



        public LPSTerm[] EvaluateNNSymbolicPostCrop(LPSState state, LPSTerm[] input)
        {
            var v = input;

            for (int i = 0; i < LayerCount; i++)
            {
                Layer curr = Layers[i];
                var stopwatch = new Stopwatch();
                stopwatch.Start();
                var w = curr.EvaluateSymbolic(state,v);
                stopwatch.Stop();
                v = w;

                Console.WriteLine("Symbolic interpreter: layer index: {0,2}, elapsed milliseconds = {1}",curr.Index,stopwatch.ElapsedMilliseconds);
            }
            return v;
        }

        public void CoalesceToVirtual()
        {
         
            List<Layer> newLayers = new List<Layer>();

            List<Layer> currAffList = new List<Layer>();

            for (int i = 0; i < LayerCount; i++)
            {
                Layer curr = Layers[i];
                if (curr.IsAffine()) 
                {
                    currAffList.Add(curr);
                    continue;
                }

                // Current layer is not affine
                // If we have anything in the affine list, we should coalesce and insert before current.
                if (currAffList.Count > 0)
                {
                    Layer virt = new VirtualLayer(currAffList);
                    currAffList.Clear();
                    newLayers.Add(virt);
                }
                newLayers.Add(curr);
            }

            if (currAffList.Count > 0)
            {
                    Layer virt = new VirtualLayer(currAffList);
                    currAffList.Clear();
                    newLayers.Add(virt);
            }

            layers_ = newLayers;
        }

    }

}
