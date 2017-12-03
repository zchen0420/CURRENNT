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

#ifndef LAYERS_GRULAYER_HPP
#define LAYERS_GRULAYER_HPP

#include "TrainableLayer.hpp"
#include "../helpers/Matrix.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a fully connected layer which uses GRU cells with 2 gates
     * and one cell per block
     *
     * weights; with P = precedingLayer().size() and L = size():
     *    ~ weights from preceding layer:
     *        - [0 .. PL-1]:    net input
     *        - [1PL .. 2PL-1]: update gate
     *        - [2PL .. 3PL-1]: reset gate
     *    ~ bias weights:
     *        - [3PL + 0  .. 3PL + L-1]:  net input
     *        - [3PL + 1L .. 3PL + 2L-1]: update gate
     *        - [3PL + 2L .. 3PL + 3L-1]: reset gate
     *    ~ internal weights (from other cells in the same layer):
     *        - [3(P+1)L + 0   .. 3(P+1)L + LL-1]:  net input
     *        - [3(P+1)L + 1LL .. 3(P+1)L + 2LL-1]: update gate
     *        - [3(P+1)L + 2LL .. 3(P+1)L + 3LL-1]: reset gate
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class GruLayer : public TrainableLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;

        struct weight_matrices_t {
            helpers::Matrix<TDevice> niInput;
            helpers::Matrix<TDevice> ugInput;
            helpers::Matrix<TDevice> rgInput;
            helpers::Matrix<TDevice> niInternal;
            helpers::Matrix<TDevice> igInternal;
            helpers::Matrix<TDevice> ugInternal;
            helpers::Matrix<TDevice> rgInternal;
        };

        struct timestep_matrices_t {
            helpers::Matrix<TDevice> tmpOutputs;
            helpers::Matrix<TDevice> tmpOutputErrors;
            helpers::Matrix<TDevice> tmpRgOutputs;
            helpers::Matrix<TDevice> niActs;
            helpers::Matrix<TDevice> ugActs;
            helpers::Matrix<TDevice> rgActs;
            helpers::Matrix<TDevice> niDeltas;
            helpers::Matrix<TDevice> ugDeltas;
            helpers::Matrix<TDevice> rgDeltas;
        };

        struct forward_backward_info_t {
            real_vector tmpOutputs;
            real_vector tmpOutputErrors;
            real_vector tmpRgOutputs;
            real_vector niActs;
            real_vector ugActs;
            real_vector rgActs;
            real_vector niDeltas;
            real_vector ugDeltas;
            real_vector rgDeltas;

            helpers::Matrix<TDevice> niActsMatrix;
            helpers::Matrix<TDevice> ugActsMatrix;
            helpers::Matrix<TDevice> rgActsMatrix;
            helpers::Matrix<TDevice> niDeltasMatrix;
            helpers::Matrix<TDevice> ugDeltasMatrix;
            helpers::Matrix<TDevice> rgDeltasMatrix;

            weight_matrices_t                weightMatrices;
            weight_matrices_t                weightUpdateMatrices;
            std::vector<timestep_matrices_t> timestepMatrices;
        };

    private:
        const bool m_isBidirectional;

        real_t *_rawNiBiasWeights;
        real_t *_rawUgBiasWeights;
        real_t *_rawRgBiasWeights;

        forward_backward_info_t m_fw;
        forward_backward_info_t m_bw;

        helpers::Matrix<TDevice> m_precLayerOutputsMatrix;

    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         * @param bidirectional  Wether the layer should be bidirectional or unidirectional
         */
        GruLayer(
            const helpers::JsonValue &layerChild,
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
            bool                      bidirectional = false
            );

        /**
         * Destructs the Layer
         */
        virtual ~GruLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * Returns true if the layer is bidirectional
         *
         * @return True if the layer is bidirectional
         */
        bool isBidirectional() const;

        /**
         * Returns the net input activations
         *
         * @return The net input activations
         */
        const real_vector& netInputActs() const;

        /**
         * Returns the net input activation deltas
         *
         * @return The net input activation deltas
         */
        const real_vector& netInputDeltas() const;

        /**
         * Returns the input gate activations
         *
         * @return The input gate activations
         */
        const real_vector& inputGateActs() const;

        /**
         * Returns the input gate deltas
         *
         * @return The input gate deltas
         */
        const real_vector& inputGateDeltas() const;

        /**
         * Returns the update gate activations
         *
         * @return The update gate activations
         */
        const real_vector& updateGateActs() const;

        /**
         * Returns the update gate deltas
         *
         * @return The update gate deltas
         */
        const real_vector& updateGateDeltas() const;

        /**
         * Returns the reset gate activations
         *
         * @return The reset gate activations
         */
        const real_vector& resetGateActs() const;

        /**
         * Returns the reset gate deltas
         *
         * @return The reset gate deltas
         */
        const real_vector& resetGateDeltas() const;

        /**
         * @see Layer::loadSequences
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();
    };

} // namespace layers


#endif // LAYERS_GRULAYER_HPP
