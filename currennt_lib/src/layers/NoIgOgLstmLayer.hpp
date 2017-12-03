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

#ifndef LAYERS_NOIGOGLSTMLAYER_HPP
#define LAYERS_NOIGOGLSTMLAYER_HPP

#include "TrainableLayer.hpp"
#include "../helpers/Matrix.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents a fully connected layer which uses LSTM cells with forget gates, peephole
     * connections and one cell per block, but no input or output gate
     *
     * weights; with P = precedingLayer().size() and L = size():
     *    ~ weights from preceding layer:
     *        - [0 .. PL-1]:    net input
     *        - [1PL .. 2PL-1]: forget gate
     *    ~ bias weights:
     *        - [2PL + 0  .. 2PL + L-1]:  net input
     *        - [2PL + 1L .. 2PL + 2L-1]: forget gate
     *    ~ internal weights (from other cells in the same layer):
     *        - [2(P+1)L + 0   .. 2(P+1)L + LL-1]:  net input
     *        - [2(P+1)L + 1LL .. 2(P+1)L + 2LL-1]: forget gate
     *    ~ peephole weights (from cell state to all gates in the same cell):
     *        - [2(P+1+L)L + 0   .. 2(P+1+L)L + L-1]:  forget gate
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class NoIgOgLstmLayer : public TrainableLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;

        struct weight_matrices_t {
            helpers::Matrix<TDevice> niInput;
            helpers::Matrix<TDevice> fgInput;
            helpers::Matrix<TDevice> niInternal;
            helpers::Matrix<TDevice> fgInternal;
        };

        struct timestep_matrices_t {
            helpers::Matrix<TDevice> tmpOutputs;
            helpers::Matrix<TDevice> tmpOutputErrors;
            helpers::Matrix<TDevice> niActs;
            helpers::Matrix<TDevice> fgActs;
            helpers::Matrix<TDevice> niDeltas;
            helpers::Matrix<TDevice> fgDeltas;
        };

        struct forward_backward_info_t {
            real_vector tmpOutputs;
            real_vector tmpOutputErrors;
            real_vector cellStates;
            real_vector cellStateErrors;
            real_vector niActs;
            real_vector fgActs;
            real_vector niDeltas;
            real_vector fgDeltas;

            helpers::Matrix<TDevice> niActsMatrix;
            helpers::Matrix<TDevice> fgActsMatrix;
            helpers::Matrix<TDevice> niDeltasMatrix;
            helpers::Matrix<TDevice> fgDeltasMatrix;

            weight_matrices_t                weightMatrices;
            weight_matrices_t                weightUpdateMatrices;
            std::vector<timestep_matrices_t> timestepMatrices;
        };

    private:
        const bool m_isBidirectional;

        real_t *_rawNiBiasWeights;
        real_t *_rawFgBiasWeights;
        real_t *_rawFgPeepholeWeights;

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
        NoIgOgLstmLayer(
            const helpers::JsonValue &layerChild,
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
            bool                      bidirectional = false
            );

        /**
         * Destructs the Layer
         */
        virtual ~NoIgOgLstmLayer();

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
         * Returns the cell states
         *
         * @return The cell states
         */
        const real_vector& cellStates() const;

        /**
         * Returns the cell state errors
         *
         * @return The cell state errors
         */
        const real_vector& cellStateErrors() const;

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
         * Returns the forget gate activations
         *
         * @return The forget gate activations
         */
        const real_vector& forgetGateActs() const;

        /**
         * Returns the forget gate deltas
         *
         * @return The forget gate deltas
         */
        const real_vector& forgetGateDeltas() const;

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


#endif // LAYERS_NOIGOGLSTMLAYER_HPP
