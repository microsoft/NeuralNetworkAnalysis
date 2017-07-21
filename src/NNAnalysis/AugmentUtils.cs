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


using System.Diagnostics;
using NNAnalysis;

namespace NNAnalysis
{

    public enum RANDTYPE
    {
        GAUSSIAN,
        UNIFORM
    }

    public interface IAugmentor
    {
        List<double[]> Augment(double[] datum);
    }

    public class AugmentBrightness : IAugmentor
    {
        ImageCoordinates _coords;
        int _how_many;
        double _max_eps;
        Random _random;
        RANDTYPE _typ;
 
        public AugmentBrightness(ImageCoordinates coords, RANDTYPE typ, int how_many, double max_eps)
        {
            _typ = typ;
            _coords = coords;
            _how_many = how_many;
            _max_eps = max_eps;
            _random = new Random(System.DateTime.Now.Millisecond);
        }

        public List<double[]> Augment(double[] datum)
        {
            List<double[]> newdatums = new List<double[]>();

            // How many to generate 
            for (int i = 0; i < _how_many; i++)
            {
                // Allocate data
                var newdatum = new double[datum.Length];
                // Sample epsilon
                double eps = (_typ == RANDTYPE.UNIFORM) ?
                    (_random.NextDouble()*2.0 - 1.0) * _max_eps:
                    Utils.URand.NextGaussian(_random) * _max_eps;

                // Add constant epsilon to all channels.
                for (int c = 0; c < _coords.ChannelCount; c++)
                {
                    for (int x = 0; x < _coords.RowCount; x++)
                    {
                        for (int y = 0; y < _coords.ColumnCount; y++)
                        {
                            int idx = _coords.GetIndex(c, x, y);
                            newdatum[idx] = Utils.UMath.Clamp(datum[idx] + eps, Utils.RobustnessOptions.MinValue, Utils.RobustnessOptions.MaxValue);
                        }
                    }
                }


                newdatums.Add(newdatum);
            }
            return newdatums;
        }
    }


    public class AugmentContrast : IAugmentor
    {
        ImageCoordinates _coords;
        int _how_many;
        Random _random;

        double _max_contrast_factor;
        double _min_contrast_factor;

        public AugmentContrast(ImageCoordinates coords, int how_many, double min_contrast_factor, double max_contrast_factor)
        {
            _coords = coords;
            _how_many = how_many;
            _max_contrast_factor = max_contrast_factor;
            _min_contrast_factor = min_contrast_factor;
            _random = new Random(System.DateTime.Now.Millisecond);


        }

        public double[] ChannelAverages(double[] datum)
        {
            double[] rets = new double[_coords.ChannelCount];

            for (int c = 0; c < _coords.ChannelCount; c++)
            {
                for (int x = 0; x < _coords.RowCount; x++)
                {
                    for (int y = 0; y < _coords.ColumnCount; y++)
                    {
                        int idx = _coords.GetIndex(c, x, y);
                        rets[c] += datum[idx];
                    }
                }
            }

            return rets;

        }


        public List<double[]> Augment(double[] datum)
        {
            List<double[]> newdatums = new List<double[]>();

            var chans = ChannelAverages(datum);

            // How many to generate 
            for (int i = 0; i < _how_many; i++)
            {
                // Allocate data
                var newdatum = new double[datum.Length];
                // Sample epsilon
                double eps = _random.NextDouble()*(_max_contrast_factor - _min_contrast_factor) + _min_contrast_factor;

                // Add constant epsilon to all channels.
                for (int c = 0; c < _coords.ChannelCount; c++)
                {
                    for (int x = 0; x < _coords.RowCount; x++)
                    {
                        for (int y = 0; y < _coords.ColumnCount; y++)
                        {
                            int idx = _coords.GetIndex(c, x, y);
                            double avgc = chans[c]/(_coords.ChannelCount * _coords.RowCount);
                            newdatum[idx] = Utils.UMath.Clamp((datum[idx] - avgc) * eps + avgc, Utils.RobustnessOptions.MinValue, Utils.RobustnessOptions.MaxValue);
                        }
                    }
                }

                newdatums.Add(newdatum);
            }
            return newdatums;
        }
    }


    public class AugmentRotation : IAugmentor
    {
        ImageCoordinates _coords;
        int _how_many;
        float _degrees;
        Random _random;

        public AugmentRotation(ImageCoordinates coords, int how_many, float degrees)
        {
            _coords = coords;
            _how_many = how_many;
            _degrees = degrees;
            Trace.Assert(degrees >= -180.0 && degrees <= 180.0);
            _random = new Random(System.DateTime.Now.Millisecond);
        }

        public List<double[]> Augment(double[] datum)
        {
            List<double[]> newdatums = new List<double[]>();

            // How many to generate 
            for (int i = 0; i < _how_many; i++)
            {
                // Allocate data
                int[] datum_int = Utils.UArray.ToIntArray(datum);


                Utils.UDraw.DisplayImageAndPause(datum_int, 32, 32, true);
                

                float eps = (float) _random.NextDouble();
                float real_agle = eps * _degrees;

                int[] newdatum_int = Utils.UDraw.Rotate(datum_int, _coords.RowCount, _coords.ColumnCount, (_coords.ChannelCount > 1), real_agle);

                Utils.UDraw.DisplayImageAndPause(newdatum_int, 32, 32, true);



                newdatums.Add(Utils.UArray.ToDoubleArray(newdatum_int));
            }

            return newdatums;
        }
    }



    public class AugmentLossyJpeg : IAugmentor
    {
        ImageCoordinates _coords;
        int _how_many;
        int _loss;  // 0L - 100L 
        Random _random;

        public AugmentLossyJpeg(ImageCoordinates coords, int how_many, int loss)
        {
            _coords = coords;
            _how_many = how_many;
            _loss = loss;
            
            Trace.Assert(loss >= 0 && loss <= 100);

            _random = new Random(System.DateTime.Now.Millisecond);
        }

        public List<double[]> Augment(double[] datum)
        {
            List<double[]> newdatums = new List<double[]>();

            // How many to generate 
            for (int i = 0; i < _how_many; i++)
            {
                // Allocate data
                int[] datum_int = Utils.UArray.ToIntArray(datum);

                Int64 photoquality = _random.Next(_loss, 101);

                int[] newdatum_int = Utils.UDraw.LossyJPGAndBack(datum_int,_coords.RowCount,_coords.ColumnCount,(_coords.ChannelCount > 1), photoquality);

                newdatums.Add(Utils.UArray.ToDoubleArray(newdatum_int));
            }

            return newdatums;
        }
    }

    public class AugmentRandom : IAugmentor
    {
        ImageCoordinates _coords;
        int _how_many;
        double _max_eps;
        Random _random;
        RANDTYPE _typ;

        public AugmentRandom(ImageCoordinates coords, RANDTYPE typ, int how_many, double eps)
        {
            _typ = typ;
            _coords = coords;
            _how_many = how_many;
            _max_eps = eps;
            _random = new Random(System.DateTime.Now.Millisecond);
        }

        public List<double[]> Augment(double[] datum)
        {
            List<double[]> newdatums = new List<double[]>();

            // How many to generate 
            for (int i = 0; i < _how_many; i++)
            {
                // Allocate data
                var newdatum = new double[datum.Length];

                // Add constant epsilon to all channels.
                for (int c = 0; c < _coords.ChannelCount; c++)
                {
                    for (int x = 0; x < _coords.RowCount; x++)
                    {
                        for (int y = 0; y < _coords.ColumnCount; y++)
                        {
                            // Sample epsilon
                            double eps = (_typ == RANDTYPE.UNIFORM) ?
                                (_random.NextDouble()*2.0 - 1.0) * _max_eps :
                                Utils.URand.NextGaussian(_random) * _max_eps;

                            int idx = _coords.GetIndex(c, x, y);
                            newdatum[idx] = Utils.UMath.Clamp(datum[idx] + eps, Utils.RobustnessOptions.MinValue, Utils.RobustnessOptions.MaxValue);
                        }
                    }
                }
                newdatums.Add(newdatum);
            }
            return newdatums;
        }
    }


    public class AugmentGeometric: IAugmentor
    {
        ImageCoordinates _coords;
        int _how_many;
        Random _random;
        int _xoffset;
        int _yoffset;
        RANDTYPE _typ;

        public AugmentGeometric(ImageCoordinates coords, RANDTYPE typ, int how_many, int xoffest, int yoffset)
        {
            _typ = typ;
            _coords = coords;
            _how_many = how_many;
            _xoffset = xoffest;
            _yoffset = yoffset;
            _random = new Random(System.DateTime.Now.Millisecond);
        }

        public List<double[]> Augment(double[] datum)
        {
            List<double[]> newdatums = new List<double[]>();

            for (int i = 0; i < _how_many; i++)
            {
                var newdatum = new double[datum.Length];

                // Sample epsilon
                double eps = (_typ == RANDTYPE.UNIFORM) ?
                    (_random.NextDouble() * 2.0 - 1.0) :
                    Utils.URand.NextGaussian(_random);

                
                for (int c = 0; c < _coords.ChannelCount; c++)
                {
                    for (int x = 0; x < _coords.RowCount; x++)
                    {
                        for (int y = 0; y < _coords.ColumnCount; y++)
                        {

                            int xnew = (int)(x + eps * _xoffset);
                            int ynew = (int)(y + eps * _yoffset);

                            if (xnew < 0 || xnew >= _coords.RowCount) xnew = x;
                            if (ynew < 0 || ynew >= _coords.ColumnCount) ynew = y;

                       
                            int idx = _coords.GetIndex(c, x, y);
                            int newidx = _coords.GetIndex(c, xnew, ynew);

                            newdatum[newidx] = datum[idx];
                        }
                    }
                }

                // Utils.UDraw.DisplayImageAndPause(Utils.UArray.ToRGBArray(datum, 1.0, 0.0), 32, 32, true);
                // Utils.UDraw.DisplayImageAndPause(Utils.UArray.ToRGBArray(newdatum, 1.0, 0.0), 32, 32, true);

                newdatums.Add(newdatum);
            }
            return newdatums;
        }
    }


    // Give instances here of Augmentor, 
    // - uniform around point within epsilon ball
    // - gaussian around point with epsilon stdev

    // - brightness: just add a fixed constant to each (R,G,B) value

    // - contrast: 
    //      R = (R - AverageR) * Contrast + AverageR           
    //      G = (G - AverageG) * Contrast + AverageG
    //      B = ...

}
