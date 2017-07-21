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
using System.Threading;
using System.Diagnostics;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NNAnalysis
{

    // Type-class trick, thanks to Claudio Russo (crusso@microsoft.com)
    public interface Num<T,V> where V : IList<T>
    {

        void AddMul(ref T tgt, T src, double d);  // tgt += src*d;
        void Add(ref T tgt, T src);               // tgt += src;
        void Add(ref T tgt, double d);            // tgt += d;
        void Mul(ref T tgt, double d);            //  tgt *= d;

        T Const(double d);

        V CreateVector(int capacity);

    }

    public struct NumInstDouble : Num<double,Vector<double>>
    {
        public void AddMul(ref double tgt,double src, double d) { tgt += src*d; }
        public void Add(ref double tgt, double src) { tgt += src;  }
        public void Mul(ref double tgt, double d) { tgt *= d; }
        public double Const(double d) { return d; }

        public Vector<double> CreateVector(int capacity) { return SparseVector.Create(capacity, 0.0); }
    }
    //public struct NumInstLPSTermVec : Num<LPSTerm,LPSTerm[]>
    //{
    //    public void AddMul(ref LPSTerm tgt, LPSTerm src, double d) { tgt.AddMul(src, d); }
    //    public void Add(ref LPSTerm tgt, LPSTerm src) { tgt.Add(src); }
    //    public void Add(ref LPSTerm tgt, double d) { tgt.Add(d);} 
    //    public void Mul(ref LPSTerm tgt, double d) { tgt.Mul(d); }
    //    public LPSTerm Const(double d) { return LPSTerm.Const(d);  }

    //    public LPSTerm[] CreateVector(int capacity)
    //    {
    //        var coeffs = DenseMatrix.Create(capacity, LPSTerm.TotalVarCount(), 0.0);
    //        var interc = DenseVector.Create(capacity, 0.0);
    //        return new LPSTerm[](coeffs, interc);
    //    }
    //}
    public struct NumInstLPSTermArr : Num<LPSTerm, LPSTerm[]>
    {
        public void AddMul(ref LPSTerm tgt, LPSTerm src, double d) { tgt.AddMul(src, d); }
        public void Add(ref LPSTerm tgt, LPSTerm src) { tgt.Add(src); }
        public void Add(ref LPSTerm tgt, double d) { tgt.Add(d); }
        public void Mul(ref LPSTerm tgt, double d) { tgt.Mul(d); }
        public LPSTerm Const(double d) { return LPSTerm.Const(d); }

        public LPSTerm[] CreateVector(int capacity)
        {
            var vec = new LPSTerm[capacity];
            for (int i = 0; i < capacity; i++)
            {
                vec[i] = LPSTerm.Const(0.0);
            }
            return vec;
        }
    }




    public class VCInfo
    {
        public ThreadLocal<Vector<double>> tempmultstorage; // = new ThreadLocal<Vector<double>>();
        public int varcount_ = 0;
        public int total_varcount_ = 0;

        public VCInfo(int total_varcount)
        {
            total_varcount_ = total_varcount;
            tempmultstorage = new ThreadLocal<Vector<double>>(() => SparseVector.Create(total_varcount,0.0));
        }
    }

    /// <summary>
    /// Representation of a linear term, like 0.3 x0 + 0.0 x1 + .... 4.3 xn + interecept_
    /// We use a Dictionary of coefficients from variable positions (0 for x0, etc).
    /// Keys without a corresponding entry are meant to have coefficient 0.0.
    /// </summary>
    public class LPSTerm 
    {

        public static int addmulcounter = 0;



        Vector<double> coefficients_;
        double intercept_;

        // Set to null to ensure someone does call InitVariableFactory() below first!
        protected static VCInfo vcinfo_ = null;


        public static LPSTerm[] IdentityMatrix(int howmany)
        {
            //Matrix<double> coeffs = DenseMatrix.CreateIdentity(howmany);
            //Vector<double> interc = DenseVector.Create(howmany, 0.0);

            // return new LPSTerm[](coeffs, interc);

            LPSTerm[] terms = new LPSTerm[howmany];
            int pos = 0;
            for (int i = 0; i < howmany; i++)
            {
                Vector<double> coeffs = SparseVector.Create(howmany, 0.0);
                coeffs[pos++] = 1.0;
                terms[i] = new LPSTerm(coeffs, 0.0);
            }
            return terms;
        }

        public static Matrix<double> UnderlyingMatrix(LPSTerm[] terms)
        {
            // Stopwatch s = new Stopwatch();
            // s.Start();

            Matrix<double> res = SparseMatrix.Create(terms.Length, LPSTerm.TotalVarCount(), 0.0);

            for (int i = 0; i < terms.Length; i++)
            {
                res.SetRow(i, terms[i].GetCoefficients());
            }

            // s.Stop();
            // Console.WriteLine("To underlying matrix: {0} milliseconds",s.ElapsedMilliseconds);

            return res;

        }


        public static Matrix<double> UnderlyingTransposeMatrix(LPSTerm[] terms)
        {

            Matrix<double> res = DenseMatrix.Create(LPSTerm.TotalVarCount(),terms.Length, 0.0);

            for (int i = 0; i < terms.Length; i++)
            {
                res.SetColumn(i, terms[i].GetCoefficients());
            }

            return res;

        }



        public static Vector<double> UnderlyingIntercept(LPSTerm[] terms)
        {
            Vector<double> intercept = DenseVector.Create(terms.Length, 0.0);
            for (int i = 0; i < terms.Length; i++)
            {
                intercept[i] = terms[i].Intercept;
            }
            return intercept;
        }

        public static LPSTerm[] FromUnderlyingAlgebra(Matrix<double> outm, Vector<double> outv)
        {
            LPSTerm[] ret = new LPSTerm[outm.RowCount];
            for (int i = 0; i < outm.RowCount; i++)
            {
                ret[i] = new LPSTerm(outm.Row(i), outv[i]);
            }
            return ret;
        }

        public static LPSTerm[] FromUnderlyingTransposeAlgebra(Matrix<double> outm, Vector<double> outv)
        {
            LPSTerm[] ret = new LPSTerm[outm.ColumnCount];
            for (int i = 0; i < outm.ColumnCount; i++)
            {
                ret[i] = new LPSTerm(outm.Column(i), outv[i]);
            }
            return ret;
        }




        public static LPSTerm[] FreshVariables(int howmany)
        {
            var tmp = new LPSTerm[howmany];
            for (int i=0; i < howmany; i++)
            {
                tmp[i] = FreshVariable();
            }
            return tmp;
        }
        private static LPSTerm FreshVariable()
        {
            var tmp = new LPSTerm();
            tmp.coefficients_[vcinfo_.varcount_] = 1.0;
            tmp.intercept_ = 0.0;
            vcinfo_.varcount_++;
            return tmp;
        }


        public static VCInfo GetVariableFactoryState()
        {
            return vcinfo_;
        }

        public static void ResetVariableFactory(int total_variables)
        {
           vcinfo_ = new VCInfo(total_variables);
           vcinfo_.varcount_ = 0;
           vcinfo_.total_varcount_ = total_variables;
        }

        public static void RestoreVariableFactory(VCInfo info)
        {
            vcinfo_ = info;
        }

        public static int TotalVarCount()
        {
            return vcinfo_.total_varcount_;
        }

        public LPSTerm(Vector<double> coeffs, double interc)
        {
            coefficients_ = coeffs;
            intercept_ = interc;
        }


        private LPSTerm()
        {
            coefficients_ = SparseVector.Create(vcinfo_.total_varcount_, 0.0);
            intercept_ = 0.0;
        }
        public int VarCount
        {
            get 
            {
                return vcinfo_.varcount_; 
            }
        }
        public void Clear()
        {
            coefficients_.Clear();
        }

        public void Densify()
        {
            coefficients_ = DenseVector.OfVector(coefficients_);
        }

        public void Sparsify()
        {
            coefficients_ = SparseVector.OfVector(coefficients_);
        }


        public Vector<double> GetCoefficients()
        {
            return coefficients_;
        }
        public double GetCoefficient(int i)
        {
            return coefficients_[i];
        }
        public void SetCoefficient(int i, double d)
        {
            coefficients_[i] = d;
        }
        public double Intercept
        {
            get { return intercept_; }
            set { intercept_ = value; }
        }
        public override String ToString()
        {
            String ret = "";
            for (int i = 0; i < vcinfo_.total_varcount_; i++)
            {
                ret += GetCoefficient(i) + "*X" + i + " + ";
            }
            ret += Intercept;

            return ret;
        }
        // this += v
        public void Add(LPSTerm v)
        {

            coefficients_.Add(v.coefficients_, coefficients_);
            intercept_ += v.intercept_;
        }

        public void Sub(LPSTerm v)
        {
            coefficients_ -= v.coefficients_;
            intercept_ -= v.intercept_;
        }

        public void Add(double d)
        {
            intercept_ += d;
        }
        // this += d*v
        public void AddMul(LPSTerm v, double d)
        {
            
            v.coefficients_.Multiply(d, vcinfo_.tempmultstorage.Value);
            coefficients_.Add(vcinfo_.tempmultstorage.Value,coefficients_);

            intercept_ += d * v.intercept_;
            addmulcounter++;
        }

        public void AddMulVec(Matrix<double> v_coeffm, Vector<double> v_intcps, Vector<double> d_vec)
        {  // this += v1*d1 + .... vn*dn

            //Matrix<double> v_coeffm = LPSTerm.UnderlyingMatrix(v);
            //Vector<double> v_intcps = LPSTerm.UnderlyingIntercept(v);
            
            var mul = d_vec * v_coeffm;

            coefficients_.Add(mul, coefficients_);

            intercept_ += v_intcps * d_vec;

        }


        public static LPSTerm Const(double d)
        {
            var v = new LPSTerm();
            v.intercept_ = d;
            return v;
        }
        public void Mul(double d)
        {
            coefficients_ *= d;
            intercept_ *= d;
        }
    }

}
