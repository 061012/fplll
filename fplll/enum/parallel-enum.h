/* Copyright (C) 2019 Marc Stevens
   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.
   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

#ifndef FPLLL_PARALLEL_ENUM_H
#define FPLLL_PARALLEL_ENUM_H

#include <atomic>
#include <fplll/defs.h>
#include <fplll/enum/enumerate_base.h>
#include <fplll/enum/enumerate_dyn.h>
#include <fplll/gso_interface.h>
#include <fplll/threadpool.h>

FPLLL_BEGIN_NAMESPACE

template <typename ZT, typename FT> class ParallelEnumerationDyn;

template <typename ZT, typename FT> class TopEvaluator : public Evaluator<FT>
{
public:
  TopEvaluator(ParallelEnumerationDyn<ZT, FT> &parent) : _parent(parent) {}

  virtual void eval_sol(const vector<FT> &new_sol_coord, const enumf &new_partial_dist,
                        enumf &maxdist)
  {
    _parent.process_top_node(new_sol_coord, new_partial_dist);
  }
  virtual void eval_sub_sol(int offset, const vector<FT> &new_sub_sol_coord, const enumf &sub_dist)
  {
  }

private:
  ParallelEnumerationDyn<ZT, FT> &_parent;
};

template <typename ZT, typename FT> class BottomEvaluator : public Evaluator<FT>
{
public:
  BottomEvaluator(ParallelEnumerationDyn<ZT, FT> &parent) : _parent(parent) {}

  virtual void eval_sol(const vector<FT> &new_sol_coord, const enumf &new_partial_dist,
                        enumf &maxdist)
  {
    _parent.process_sol(new_sol_coord, new_partial_dist, maxdist, this->normExp);
  }
  virtual void eval_sub_sol(int offset, const vector<FT> &new_sub_sol_coord, const enumf &sub_dist)
  {
    _parent.process_subsol(offset, new_sub_sol_coord, sub_dist, this->normExp);
  }

private:
  ParallelEnumerationDyn<ZT, FT> &_parent;
};

template <typename ZT, typename FT> class ParallelEnumerationDyn
{
public:
  ParallelEnumerationDyn(MatGSOInterface<ZT, FT> &gso, Evaluator<FT> &evaluator)
      : _topeval(*this), _topenum(gso, _topeval), _gso(gso), _evaluator(evaluator)
  {
  }

  void enumerate(int first, int last, FT &fmaxdist, long fmaxdistexpo, int split = -1,
                 const vector<FT> &target_coord = vector<FT>(),
                 const vector<enumxt> &subtree  = vector<enumxt>(),
                 const vector<enumf> &pruning   = vector<enumf>());

  inline uint64_t get_nodes() const { return _nodes; }

  inline void process_top_node(const vector<FT> &new_sol_coord, const enumf &new_partial_dist)
  {
    while (true)
    {
      {
        lock_guard lock(_mutex);
        if (_toptrees.size() < _toptrees.capacity())
        {
          _toptrees.emplace_back(new_sol_coord.size());
          for (unsigned i = 0; i < new_sol_coord.size(); ++i)
            _toptrees.back()[i] = new_sol_coord[i].get_d();
          return;
        }
      }
      // no place, so lets work and then try again
      do_work(_bottom_enums.size() - 1);
    }
  }

  inline void process_sol(const vector<FT> &new_sol_coord, const enumf &new_partial_dist,
                          enumf &maxdist, long norm_exp)
  {
    lock_guard lock(_mutex);
    if (_maxdist < 0)
      _maxdist = maxdist;
    if (_evaluator.normExp != norm_exp)
      _evaluator.set_normexp(norm_exp);
    _evaluator.eval_sol(new_sol_coord, new_partial_dist, _maxdist);
    maxdist = _maxdist;
  }

  inline void process_subsol(int offset, const vector<FT> &new_sub_sol_coord, const enumf &sub_dist,
                             long norm_exp)
  {
    lock_guard lock(_mutex);
    if (_evaluator.normExp != norm_exp)
      _evaluator.set_normexp(norm_exp);
    _evaluator.eval_sub_sol(offset, new_sub_sol_coord, sub_dist);
  }

  bool do_work(unsigned i);
  void thread_job(unsigned i);

private:
  mutex _mutex;
  int _threads;

  int _first, _last, _split, _d;
  FT _fmaxdist;
  long _fmaxdistexpo;
  vector<FT> _target_coord;
  //  vector<enumxt> _subtree;
  vector<enumf> _pruning;
  bool _dual;
  std::atomic_uint64_t _nodes;

  TopEvaluator<ZT, FT> _topeval;
  EnumerationDyn<ZT, FT> _topenum;
  std::vector<BottomEvaluator<ZT, FT>> _bottom_evals;
  std::vector<EnumerationDyn<ZT, FT>> _bottom_enums;
  std::vector<FT> _bottom_fmaxdist;
  std::vector<vector<enumxt>> _toptrees;

  MatGSOInterface<ZT, FT> &_gso;
  Evaluator<FT> &_evaluator;
  enumf _maxdist;

  std::atomic_bool _finished;
};

FPLLL_END_NAMESPACE

#endif
