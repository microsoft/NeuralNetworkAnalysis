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

using System.IO;

namespace NNAnalysis
{

    public interface Accessor<T>
    {
        T Get();
    }
    public class MemAccessor<T> : Accessor<T>
    {
        T data_;
        public MemAccessor(T data)
        {
            data_ = data;
        }
        public T Get() { return data_; }
    }

    public class BinDataAccessor: Accessor<double[]>
    {
        String fn_;
        
        public BinDataAccessor(String fn)
        {
            fn_ = fn;
        }

        // NB: 1 byte becomes 1 double here! I.e. we were using 0-255 levels.
        public double[] Get()
        {
            byte[] bytes = File.ReadAllBytes(fn_);
            return Utils.UArray.ToDoubleArrayFromDoubleBytes(bytes);
        }
    }
    public class TxtDataAccessor : Accessor<int>
    {
        int datum;

        public TxtDataAccessor(String fn)
        {
            String rawstr = File.ReadAllText(fn);
            datum = int.Parse(rawstr);
        }

        public int Get()
        {
            return datum;
        }
    }



    public class DSList<T> 
    {
        List<Accessor<T>> elems_;
        public DSList()
        {
            elems_ = new List<Accessor<T>>();
        }

        public DSList(List<Accessor<T>> elems)
        {
            elems_ = elems;
        }

        public T Get(int index)
        {
            return elems_[index].Get();
        }
        public Accessor<T> GetAccessor(int index)
        {
            return elems_[index];
        }
        public int Count()
        {
            return elems_.Count();
        }

        public void Add(Accessor<T> datum)
        {
            elems_.Add(datum);
        }
        public void Set(int index, Accessor<T> datum)
        {
            elems_[index] = datum;
        }
        public DSList<T> CreatePermutation(int[] permutation)
        {
            DSList<T> ret = new DSList<T>();
            for (int i = 0; i < Count(); i++)
            {
                ret.elems_.Add(elems_[permutation[i]]);
            }
            return ret;
        }
        public List<DSList<T>> CreateSplit(int[] counts)
        {
            List<DSList<T>> result = new List<DSList<T>>();

            int start = 0;
            for (int i = 0; i < counts.Length; i++)
            {
                DSList<T> data = new DSList<T>();
                for (int j = 0; j < counts[i]; j++)
                {
                    data.Add(GetAccessor(start+j)); 
                }
                result.Add(data);
                start += counts[i];
            }
            if (start != Count())
            {
                throw new Exception("Inconsistent dimensions!");
            }
            return result;
        }
        public Tuple<DSList<T>, DSList<T>> CreateSplit(int count)
        {
            var counts = new int[2];
            counts[0] = count;
            counts[1] = Count() - count;
            if (counts[1] < 0) throw new Exception("Cant split past end of dataset!");

            var splits = CreateSplit(counts);

            return new Tuple<DSList<T>, DSList<T>>(splits[0], splits[1]);

        }
    }

 
    /// <summary>
    /// Image-agnostic representation of a Dataset.
    /// Contains the data, their labels and the number of labels.
    /// Image-metadata exist in the ImageDataset wrapper.
    /// </summary>
    public class Dataset
    {
        DSList<double[]> data_; // The data collection
        DSList<int> labels_;    // The labels collection
        // Invariant: data_.Length == labels_.Length
        int labelCount_;        // How many classes of labels exist? (e.g. 10 in CIFAR-10)


        public bool HasUninitialized()
        {
            // exists X in dataset. forall i. X(i) == 0
            for (int i = 0; i < data_.Count(); i++)
            {
                double[] datum = data_.Get(i);
                
                bool allzero = true;
                for (int j=0; j < datum.Length; j++)
                {
                    if (datum[j] != 0.0)
                    {
                        allzero = false;
                        break;
                    }
                }

                if (allzero) return true;
            }
            return false;
        }



        public Dataset(DSList<double[]> data, DSList<int> labels, int labelCount)
        {
            data_ = data;
            labels_ = labels;
            labelCount_ = labelCount;
        }

        public Dataset(List<double[]> images, List<int> labels, int labelCount)
        {
            var ds = images.ConvertAll(x => (Accessor<double[]>) new MemAccessor<double[]>(x));
            var ls = labels.ConvertAll(x => (Accessor<int>) new MemAccessor<int>(x));
            data_ = new DSList<double[]>(ds);
            labels_ = new DSList<int>(ls);
            labelCount_ = labelCount;
        }






        public Dataset(String directory, int labelCount)
        {
            Console.WriteLine("Loading directory-based dataset.");
            Console.WriteLine("NB: (a) expecting a 'coordinates' text file (3 lines, height, width, channel)");
            Console.WriteLine("    (b) expecting a set of <id>.datum binary and <id>.label text file containing label");

            DirectoryInfo d = new DirectoryInfo(directory);            

            data_ = new DSList<double[]>();
            labels_ = new DSList<int>();

            foreach (var file in d.GetFiles("*.datum"))
            {
                var fa = new BinDataAccessor(file.FullName);
                string labelfn = System.IO.Path.ChangeExtension(file.FullName, ".label");
                var fl = new TxtDataAccessor(labelfn);

                data_.Add(fa);
                labels_.Add(fl);
            }

            labelCount_ = labelCount;
        }

        public Dataset(Dataset d)
        {
            data_ = new DSList<double[]>();
            labels_ = new DSList<int>();

            for (int i =0; i < d.Count(); i++)
            {
                data_.Add(d.data_.GetAccessor(i));
                labels_.Add(d.labels_.GetAccessor(i));
            }
            labelCount_ = d.LabelCount();

        }

        public Dataset(int labelcount)
        {
            data_ = new DSList<double[]>();
            labels_ = new DSList<int>();
            labelCount_ = labelcount;
        }

        public int LabelCount()
        {
            return labelCount_;
        }
        public int Count()
        {
            return data_.Count();
        }
        public int Dimension()
        {
            return data_.Get(0).Length;
        }
        public DSList<double[]> Data
        {
            get { return data_; }
        }
        public DSList<int> Labels
        {
            get { return labels_; }
        }

        public double[] GetDatum(int i)
        {
            return data_.Get(i);
        }
        public int GetLabel(int i)
        {
            return labels_.Get(i);
        }

        double[] MeanDatum()
        {
            int count = Count();
            int dim = Dimension();
            double[] ret = new double[dim];

            for (int i=0; i< count; i++)
            {
                for (int j=0; j<dim; j++)
                {
                    ret[j] += (double)data_.Get(i)[j];
                }
            }

            for (int j = 0; j < dim; j++)
            {
                ret[j] /= count;
            }
            return ret;
        }


        // Random (Fisher-Yates) permutation of a dataset
        public Dataset CreateShuffle(Random random)
        {
            int[] permutation = Utils.URand.NextPermutation(random, Count());

            Dataset d = new Dataset(
                data_.CreatePermutation(permutation), 
                labels_.CreatePermutation(permutation),
                LabelCount());

            return d;
        }

        public void Update(IEnumerable<Tuple<double[], int>> points)
        {
            foreach (var tup in points)
            {
                data_.Add(new MemAccessor<double[]>(tup.Item1));
                labels_.Add(new MemAccessor<int>(tup.Item2));
            }
        }
        
        // Add points with the same label
        public void Update(IEnumerable<double[]> points, int label)
        {
            foreach (var dat in points)
            {
                data_.Add(new MemAccessor<double[]>(dat));
                labels_.Add(new MemAccessor<int>(label));
            }

        }

        public List<Dataset> CreateSplit(int[] counts)
        {
            var datasplits = data_.CreateSplit(counts);
            var labelsplit = labels_.CreateSplit(counts);

            List<Dataset> d = new List<Dataset>();

            for (int i = 0; i < datasplits.Count(); i++)
            {
                d.Add(new Dataset(datasplits[i],labelsplit[i],labelCount_));
            }
            return d;
        }
        public Tuple<Dataset, Dataset> CreateSplit(int count)
        {
            if (count > Count())
            {
                    throw new Exception("Split point can't be after the dataset!");
            }

            var split = CreateSplit(new int[] { count, Count() - count });
            return new Tuple<Dataset, Dataset>(split[0], split[1]);
        }

        public void Union(params Dataset[] datasets)
        {
            UnionMany(datasets);
        }

        public void UnionMany(Dataset[] datasets)
        {
            foreach (Dataset dataset in datasets)
            {
                for (int i = 0; i < dataset.Count(); i++)
                {
                    data_.Add(dataset.data_.GetAccessor(i)); 
                    labels_.Add(dataset.labels_.GetAccessor(i));
                }

            }
        }

    }




    /// <summary>
    ///  Wrapper around a Dataset. Contains image encoding information
    /// </summary>
    public class ImageDataset
    {
        Dataset dataset_;
        int channelCount_;
        int rowCount_;
        int columnCount_;
        bool isColor_;

        public ImageDataset(String directory, int labelcount)
        {
            string[] coords = File.ReadAllLines(directory + "\\" + "coordinates");
            rowCount_ = int.Parse(coords[0]);
            columnCount_ = int.Parse(coords[1]);
            channelCount_ = int.Parse(coords[2]);
            isColor_ = (channelCount_ > 1);
            dataset_ = new Dataset(directory, labelcount);
        }


        public ImageDataset(Dataset dataset, int channelCount, int rowCount, int columnCount, bool isColor)
        {
            dataset_ = dataset;
            channelCount_ = channelCount;
            rowCount_ = rowCount;
            columnCount_ = columnCount;
            isColor_ = isColor;
        }
        public ImageDataset(Dataset dataset, Tuple<int, int, int, bool> metadata) : 
            this(dataset, metadata.Item1, metadata.Item2, metadata.Item3, metadata.Item4) { }

        public Dataset Dataset
        {
            get { return dataset_; }
            set { dataset_ = value; }
        }
        public int ChannelCount
        {
            get { return channelCount_; }
        }
        public int RowCount
        {
            get { return rowCount_; }
        }
        public int ColumnCount
        {
            get { return columnCount_; }
        }
        public bool IsColor
        {
            get { return isColor_; }
        }
        public Tuple<int, int, int, bool> Metadata
        {
            get { return new Tuple<int, int, int, bool>(channelCount_, rowCount_, columnCount_, isColor_); }
        }

        public Tuple<ImageDataset, ImageDataset> Split(int count)
            {
                var split = this.Dataset.CreateSplit(count);
                var id1 = new ImageDataset(split.Item1, this.Metadata);
                var id2 = new ImageDataset(split.Item2, this.Metadata);
                return new Tuple<ImageDataset, ImageDataset>(id1,id2);
            }
        public List<ImageDataset> ShuffleSplitMany(int counts)
        {
            int buckets = (int)Math.Ceiling((double)this.Dataset.Count() / (double)counts);
            int[] clist = new int[buckets];
            for (int i = 0; i < buckets; i++)
            {
                clist[i] = Math.Min(counts, this.Dataset.Count() - i * counts);
            }

            var splits = this.Dataset.CreateShuffle(new Random()).CreateSplit(clist);

            List<ImageDataset> ret = new List<ImageDataset>(splits.Count());

            for (int i = 0; i < splits.Count(); i++)
            {
                ret.Add(new ImageDataset(splits[i], this.Metadata));
            }

            return ret;

        }
        public void Update(IEnumerable<Tuple<double[], int>> newImages)
        {
            this.Dataset.Update(newImages);
        }

        public void Update(IEnumerable<double[]> newImages, int newLabel)
        {
            this.Dataset.Update(newImages, newLabel);
        }


    }

    namespace Utils
    {

        public static class Data
        {
            public const int NO_LABEL = -1;
            public const int ANY_LABEL = -2;

            public static bool LabelMatch(int label1, int label2)
            {
                return label1 == ANY_LABEL || label2 == ANY_LABEL || label1 == label2;
            }

            public static void CalculateDistances(Dataset dataset)
            {

                double[,] avgdata = new double[dataset.LabelCount(), dataset.LabelCount()];
                int[,] countdata  = new int[dataset.LabelCount(), dataset.LabelCount()];

                List<Tuple<double[], int>> data = new List<Tuple<double[], int>>();

                for (int i = 0; i < dataset.Count(); i++)
                {
                    data.Add(new Tuple<double[], int>(dataset.GetDatum(i), dataset.GetLabel(i)));
                }

                var query = data.GroupBy(x => x.Item2);

                foreach (IGrouping<int, Tuple<double[], int>> grp1 in query)
                {
                    foreach (IGrouping<int,Tuple<double[],int>> grp2 in query)
                    {
                        if (grp1 == grp2) continue;        
                        foreach (var x1 in grp1)
                        {
                            foreach (var x2 in grp2)
                            {
                                double dist = UMath.L1Distance(x1.Item1, x2.Item1);
                                avgdata[x1.Item2, x2.Item2] += dist;
                                countdata[x1.Item2, x2.Item2]++;
                            }
                        }
                    }
                }

                for (int i = 0; i < dataset.LabelCount(); i++)
                {
                    for (int j = 0; j < dataset.LabelCount(); j++)
                    {
                        if (i == j) continue;
                        avgdata[i, j] = avgdata[i, j] / (double)countdata[i, j];
                    }
                }
                Console.WriteLine("Distance statistics:");

                for (int i = 0; i < dataset.LabelCount(); i++)
                {
                    for (int j = 0; j < dataset.LabelCount(); j++)
                    {
                        if (i == j) continue;
                        Console.WriteLine("Average distance, classes {0}-{1} = {2}", i, j, avgdata[i, j]);
                    }
                }

            }


            public static ImageDataset UnionMany(List<ImageDataset> imagesets)
            {
                var metadata = imagesets[0].Metadata;
                Dataset[] dlist = new Dataset[imagesets.Count()];
                for (int i = 0; i < imagesets.Count(); i++)
                {
                    dlist[i] = imagesets[i].Dataset;
                }

                Dataset dataset = new Dataset(new DSList<double[]>(), new DSList<int>(), imagesets[0].Dataset.LabelCount());
                dataset.UnionMany(dlist);

                return new ImageDataset(dataset, metadata);

            }

 
        }
    }

}
