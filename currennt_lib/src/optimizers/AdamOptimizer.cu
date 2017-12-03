/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "AdamOptimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct UpdateMsFn
    {
        real_t beta1;
        const real_t *ms;
        const real_t *gs;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            return (1 - beta1) * gs[weightIdx] + beta1 * ms[weightIdx];
        }
    };

    struct UpdateVsFn
    {
        real_t beta2;
        const real_t *vs;
        const real_t *gs;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            real_t g = gs[weightIdx];
            return (1 - beta2) * (g*g) + beta2 * vs[weightIdx];
        }
    };

    struct UpdateWeightFn
    {
        real_t learningRate;
        real_t beta1;
        real_t beta2;
        real_t epsilon;

        const real_t *weights;
        const real_t *ms;
        const real_t *vs;
        real_t *weightDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            // calculate and store the weight delta
            real_t m = ms[weightIdx] / (1 - beta1);
            real_t v = vs[weightIdx] / (1 - beta2);
            real_t delta = - learningRate * m / (sqrt(v) + epsilon);

            weightDeltas[weightIdx] = delta;

            // update weight
            return weights[weightIdx] + delta;
        }
    };

} // anonymous namespace
} // namespace internal


namespace optimizers {

    template <typename TDevice>
    void AdamOptimizer<TDevice>::_updateWeights()
    {
        internal::UpdateMsFn updateMsFn;
        internal::UpdateVsFn updateVsFn;
        internal::UpdateWeightFn updateWeightFn;
        updateWeightFn.learningRate = m_learningRate;
        updateWeightFn.beta1 = updateMsFn.beta1 = m_beta1;
        updateWeightFn.beta2 = updateVsFn.beta2 = m_beta2;
        updateWeightFn.epsilon = m_epsilon;

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());

            if (!layer) continue;

            {
                updateMsFn.gs = helpers::getRawPointer(this->_curWeightUpdates()[i]);
                updateMsFn.ms = helpers::getRawPointer(m_ms[i]);
                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    m_ms[i].begin(),
                    updateMsFn
                    );
            }
            {
                updateVsFn.gs = updateMsFn.gs;
                updateVsFn.vs = helpers::getRawPointer(m_vs[i]);
                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    m_vs[i].begin(),
                    updateVsFn
                    );
            }
            {
                updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
                updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);
                updateWeightFn.ms            = helpers::getRawPointer(m_ms[i]);
                updateWeightFn.vs            = helpers::getRawPointer(m_vs[i]);
                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    layer->weights().begin(),
                    updateWeightFn
                    );
            }
        }
    }

    template <typename TDevice>
    AdamOptimizer<TDevice>::AdamOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet,
        data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery,
        real_t learningRate, real_t beta1, real_t beta2, real_t epsilon)
        : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
        , m_learningRate    (learningRate)
        , m_beta1           (beta1)
        , m_beta2           (beta2)
        , m_epsilon         (epsilon)
    {
        // intialize the weight deltas vectors with zeros
        m_weightDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_weightDeltas.size(); ++i)
            thrust::fill(m_weightDeltas[i].begin(), m_weightDeltas[i].end(), 0);

        m_ms = this->_curWeightUpdates();
        for (size_t i = 0; i < m_ms.size(); ++i)
            thrust::fill(m_ms[i].begin(), m_ms[i].end(), 0);

        m_vs = this->_curWeightUpdates();
        for (size_t i = 0; i < m_vs.size(); ++i)
            thrust::fill(m_vs[i].begin(), m_vs[i].end(), 0);
    }

    template <typename TDevice>
    AdamOptimizer<TDevice>::~AdamOptimizer()
    {
    }

    template <typename TDevice>
    void AdamOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        Optimizer<TDevice>::exportState(jsonDoc);

        Optimizer<TDevice>::_exportWeights(jsonDoc, "adaptive_moment_estimation_optimizer_grad_first_expectation", m_ms);
        Optimizer<TDevice>::_exportWeights(jsonDoc, "adaptive_moment_estimation_optimizer_grad_second_expectation", m_vs);
        Optimizer<TDevice>::_exportWeights(jsonDoc, "adaptive_moment_estimation_optimizer_weight_deltas", m_weightDeltas);
    }

    template <typename TDevice>
    void AdamOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importState(jsonDoc);

        Optimizer<TDevice>::_importWeights(jsonDoc, "adaptive_moment_estimation_optimizer_grad_first_expectation", &m_ms);
        Optimizer<TDevice>::_importWeights(jsonDoc, "adaptive_moment_estimation_optimizer_grad_second_expectation", &m_vs);
        Optimizer<TDevice>::_importWeights(jsonDoc, "adaptive_moment_estimation_optimizer_weight_deltas", &m_weightDeltas);
    }

    // explicit template instantiations
    template class AdamOptimizer<Cpu>;
    template class AdamOptimizer<Gpu>;

} // namespace optimizers
