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

#include "AdaDeltaOptimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct UpdateSSFn
    {
        real_t rho;
        const real_t *notSquareds;
        const real_t *sumOfSquareds;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            real_t ns = notSquareds[weightIdx];
            return (1 - rho) * ns * ns + rho * sumOfSquareds[weightIdx];
        }
    };

    struct UpdateWeightFn
    {
        real_t epsilon;

        const real_t *weights;
        const real_t *weightUpdates;
        const real_t *ssGrads;
        const real_t *ssDeltas;
        real_t *weightDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            // calculate and store the weight delta
            real_t delta = - sqrt(ssDeltas[weightIdx] + epsilon) / sqrt(ssGrads[weightIdx] + epsilon) * weightUpdates[weightIdx];
            weightDeltas[weightIdx] = delta;

            // update weight
            return weights[weightIdx] + delta;
        }
    };

} // anonymous namespace
} // namespace internal


namespace optimizers {

    template <typename TDevice>
    void AdaDeltaOptimizer<TDevice>::_updateWeights()
    {
        internal::UpdateSSFn updateSSFn;
        updateSSFn.rho = m_rho;
        internal::UpdateWeightFn updateWeightFn;
        updateWeightFn.epsilon = m_epsilon;

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (!layer)
                continue;

            {
                updateSSFn.notSquareds   = helpers::getRawPointer(this->_curWeightUpdates()[i]);
                updateSSFn.sumOfSquareds = helpers::getRawPointer(m_ssGrads[i]);
                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    m_ssGrads[i].begin(),
                    updateSSFn
                    );
            }

            {
                updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
                updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[i]);
                updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);
                updateWeightFn.ssGrads       = helpers::getRawPointer(m_ssGrads[i]);
                updateWeightFn.ssDeltas      = helpers::getRawPointer(m_ssDeltas[i]);
                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    layer->weights().begin(),
                    updateWeightFn
                    );
            }

            {
                updateSSFn.notSquareds   = helpers::getRawPointer(m_weightDeltas[i]);
                updateSSFn.sumOfSquareds = helpers::getRawPointer(m_ssDeltas[i]);
                thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>((int)layer->weights().size()),
                    m_ssDeltas[i].begin(),
                    updateSSFn
                    );
            }
        }
    }

    template <typename TDevice>
    AdaDeltaOptimizer<TDevice>::AdaDeltaOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet,
        data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery,
        real_t rho, real_t epsilon)
        : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
        , m_rho             (rho)
        , m_epsilon         (epsilon)
    {
        // intialize the weight deltas vectors with zeros
        m_weightDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_weightDeltas.size(); ++i)
            thrust::fill(m_weightDeltas[i].begin(), m_weightDeltas[i].end(), 0);

        m_ssGrads = this->_curWeightUpdates();
        for (size_t i = 0; i < m_ssGrads.size(); ++i)
            thrust::fill(m_ssGrads[i].begin(), m_ssGrads[i].end(), 0);

        m_ssDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_ssDeltas.size(); ++i)
            thrust::fill(m_ssDeltas[i].begin(), m_ssDeltas[i].end(), 0);
    }

    template <typename TDevice>
    AdaDeltaOptimizer<TDevice>::~AdaDeltaOptimizer()
    {
    }

    template <typename TDevice>
    void AdaDeltaOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        Optimizer<TDevice>::exportState(jsonDoc);

        Optimizer<TDevice>::_exportWeights(jsonDoc, "adaptive_delta_optimizer_squrad_grad_expectation", m_ssGrads);
        Optimizer<TDevice>::_exportWeights(jsonDoc, "adaptive_delta_optimizer_squrad_delta_expectation", m_ssDeltas);
        Optimizer<TDevice>::_exportWeights(jsonDoc, "adaptive_delta_optimizer_weight_deltas", m_weightDeltas);
    }

    template <typename TDevice>
    void AdaDeltaOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importState(jsonDoc);

        Optimizer<TDevice>::_importWeights(jsonDoc, "adaptive_delta_optimizer_squared_grad_expectation", &m_ssGrads);
        Optimizer<TDevice>::_importWeights(jsonDoc, "adaptive_delta_optimizer_squared_delta_expectation", &m_ssDeltas);
        Optimizer<TDevice>::_importWeights(jsonDoc, "adaptive_delta_optimizer_weight_deltas", &m_weightDeltas);
    }

    // explicit template instantiations
    template class AdaDeltaOptimizer<Cpu>;
    template class AdaDeltaOptimizer<Gpu>;

} // namespace optimizers
