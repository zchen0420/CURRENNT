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

#include "AdaGradOptimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct UpdateWeightFn
    {
        real_t learningRate;
        real_t epsilon;

        const real_t *weights;
        const real_t *weightUpdates;
        real_t       *ssGrads;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            // calculate hist
            real_t squaredGrad = weightUpdates[weightIdx];
            ssGrads[weightIdx] += squaredGrad * squaredGrad;

            // calculate and store the weight delta
            real_t delta = - learningRate / sqrt(ssGrads[weightIdx] + epsilon) * weightUpdates[weightIdx];

            // calculate the new weight
            real_t newWeight = weights[weightIdx] + delta;

            return newWeight;
        }
    };

} // anonymous namespace
} // namespace internal


namespace optimizers {

    template <typename TDevice>
    void AdaGradOptimizer<TDevice>::_updateWeights()
    {
        internal::UpdateWeightFn updateWeightFn;
        updateWeightFn.epsilon      = m_epsilon;

        for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
            layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
            if (!layer)
                continue;

            updateWeightFn.learningRate = m_learningRate;
            if (layer->learningRate() >= 0.0)
                updateWeightFn.learningRate = layer->learningRate();

            {
                updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
                updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[i]);
                updateWeightFn.ssGrads       = helpers::getRawPointer(m_ssGrads[i]);
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
    AdaGradOptimizer<TDevice>::AdaGradOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet,
        data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery, 
        real_t learningRate, real_t epsilon)
        : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
        , m_learningRate    (learningRate)
        , m_epsilon         (epsilon)
    {
        // intialize the weight deltas vectors with zeros
        m_ssGrads = this->_curWeightUpdates();
        for (size_t i = 0; i < m_ssGrads.size(); ++i)
            thrust::fill(m_ssGrads[i].begin(), m_ssGrads[i].end(), 0);
    }

    template <typename TDevice>
    AdaGradOptimizer<TDevice>::~AdaGradOptimizer()
    {
    }

    template <typename TDevice>
    void AdaGradOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        Optimizer<TDevice>::exportState(jsonDoc);

        Optimizer<TDevice>::_exportWeights(jsonDoc, "adaptive_gradient_optimizer_weight_deltas", m_ssGrads);
    }

    template <typename TDevice>
    void AdaGradOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importState(jsonDoc);

        Optimizer<TDevice>::_importWeights(jsonDoc, "adaptive_gradient_optimizer_weight_deltas", &m_ssGrads);
    }

    // explicit template instantiations
    template class AdaGradOptimizer<Cpu>;
    template class AdaGradOptimizer<Gpu>;

} // namespace optimizers
