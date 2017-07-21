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

using Microsoft.SolverFoundation.Common;
using Microsoft.SolverFoundation.Services;
using SolverFoundation.Plugin.Gurobi;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics;

namespace NNAnalysis
{
    namespace Utils
    {
        public class LPSolver
        {

            GurobiSolver solver_;
            int[] vars_;
            int input_dimension_;

            int objective_id; 

            int ct_cnt = 0;

            public LPSolver( 
                int input_dimension, 
                int total_constraint_count,
                double[] origin,    // Just the image, not the epsilon
                double originbound  // Bounding rectangle
                )
            {
                
                solver_ = new GurobiSolver();

                input_dimension_ = input_dimension;

                int varCount = LPSTerm.TotalVarCount();

                Console.WriteLine("Number of variables: " + varCount);
                vars_ = new int[varCount];
                for (int i = 0; i < varCount; i++)
                {
                    int vid;
                    solver_.AddVariable("x" + i, out vid);
                    solver_.SetIntegrality(vid, RobustnessOptions.Integrality);
                    if (i < origin.Length)
                    {
                        double lb = Math.Max(Utils.RobustnessOptions.MinValue, origin[i] - originbound);
                        double ub = Math.Min(Utils.RobustnessOptions.MaxValue, origin[i] + originbound);

                        if (lb <= ub)
                        {
                            // Tighter bounds for the image variables!
                            solver_.SetBounds(vid, lb, ub);
                        }
                        else
                        {
                            // Bound validation failed, very weird. Oh well just don't use the bounds. 
                            // The programmer got the Min/Max values wrong.
                            solver_.SetBounds(vid,origin[i] - originbound,origin[i] + originbound);
                        }
                    }
                    else
                    {
                        solver_.SetBounds(vid, Utils.RobustnessOptions.MinValue, Utils.RobustnessOptions.MaxValue);
                    }

                    vars_[i] = vid;
                }
            }

            public void AddConstraint(LPSConstraint ct)
            {
                int ctid = ct_cnt;

                solver_.AddRow("constraint" + ct_cnt, out ctid);
                Vector<double> coefficients = ct.Term.GetCoefficients();
                int totalvars = LPSTerm.TotalVarCount();

                for (int j = 0; j < totalvars; j++)
                {
                    // Due to the way MSF works, if we are adding a 0 coefficient
                    // this amounts to actually removing it. However, the coefficient
                    // is not there to start with, hence let's not add it, at all! 
                    if (coefficients[j] != 0) 
                        solver_.SetCoefficient(ctid, vars_[j], coefficients[j]);
                }

                switch (ct.Inequality)
                {
                    case InequalityType.LT:
                        solver_.SetUpperBound(ctid, -ct.Term.Intercept); // - RobustnessOptions.StrictInequalityLambda * Math.Abs(ct.Term.Intercept));
                        break;
                    case InequalityType.LE:
                        solver_.SetUpperBound(ctid, -ct.Term.Intercept);
                        break;
                    case InequalityType.GT:
                        solver_.SetLowerBound(ctid, -ct.Term.Intercept); // + RobustnessOptions.StrictInequalityLambda * Math.Abs(ct.Term.Intercept));
                        break;
                    case InequalityType.GE:
                        solver_.SetLowerBound(ctid, -ct.Term.Intercept);
                        break;
                    case InequalityType.EQ:
                        // solver_.SetValue(ctid, -ct.Term.Intercept); WRONG
                        solver_.SetBounds(ctid, -ct.Term.Intercept, -ct.Term.Intercept);
                        break;
                    default:
                        break;
                }
                ct_cnt++;

            }

            public void AddConstraints(LPSConstraints constraints, Nullable<LPSObjective> objective)
            {
                // Constraints
                int numConstraints = constraints.Count;
                int tmp = 0;
                Console.WriteLine("LP constraints: " + numConstraints);
                int varCount = LPSTerm.TotalVarCount();
                foreach (LPSConstraint ct in constraints)
                {
                    AddConstraint(ct);
                    tmp++;
                    // Console.Write("\rAdding LP constraints: {0:0.000}%", (double)tmp * 100.0 / numConstraints);
                }
                Console.WriteLine();

                if (objective.HasValue)
                {
                    int objid;
                    solver_.AddRow("Objective", out objid);

                    for (int j = 0; j < varCount; j++)
                    {
                        solver_.SetCoefficient(objid, vars_[j], objective.Value.term.GetCoefficient(j));
                        // objConstr += objective.Value.term.GetCoefficient(j) * vars[j];
                    }

                    switch (objective.Value.type)
                    {
                        case LPSObjectiveType.Max:
                            solver_.AddGoal(objid, 10, false);
                            objective_id = objid;
                            break;
                        case LPSObjectiveType.Min:
                            solver_.AddGoal(objid, 10, true);
                            objective_id = objid;
                            break;
                    }
                }

            }

            public double[] SolveLowLevelLP()
            {
                // Solve the LP
                Console.Write("Solving LP ... ");

                GurobiParams pms = new GurobiParams();
                pms.OutputFlag = false;
                pms.TimeLimit = (int)RobustnessOptions.LPTimeMilliSeconds;

                // Try to prevent GC from happening here ...
                // First do a massive reclaim ... 
                GC.Collect(2);
                // Then save the old GC mode and set the one now to low latency ... 
                var old_gc_mode = System.Runtime.GCSettings.LatencyMode;
                System.Runtime.GCSettings.LatencyMode = System.Runtime.GCLatencyMode.LowLatency;

                var answer = solver_.Solve(pms);

                // Restore GC mode ... 
                System.Runtime.GCSettings.LatencyMode = old_gc_mode; 

                // DV: For now!
                //double solval = answer.GetSolutionValue(objective_id).ToDouble();
                //Console.WriteLine("Objective (row) value: {0}", solval);
                //Console.WriteLine("Objective (variable) GetValue: {0}", answer.GetValue(vars_[LPSTerm.TotalVarCount() - 1]).ToDouble());
                //var report = solver_.GetReport(LinearSolverReportType.None);
                //Console.WriteLine("Report:");
                //Console.WriteLine(report);
                          

                LinearResult result = answer.LpResult;

                if (result != LinearResult.Optimal)
                {
                    if (result != LinearResult.Feasible)
                    {
                        Console.WriteLine("LP non-feasible");
                        return null;
                    }
                    else // Feasible
                    {
                        Console.WriteLine("LP feasible but non-optimal solution");
                    }
                }
                Console.WriteLine("LP optimal solution found");

                double[] vs = new double[input_dimension_];
                for (int i = 0; i < input_dimension_; i++)
                {
                    vs[i] = answer.GetValue(vars_[i]).ToDouble();
                }

                return vs;
            }

        }

    }
}
