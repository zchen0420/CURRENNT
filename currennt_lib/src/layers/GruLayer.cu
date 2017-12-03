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

#include "GruLayer.hpp"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    typedef activation_functions::Logistic gate_act_fn_t;
    typedef activation_functions::Tanh     cell_input_act_fn_t;
    typedef activation_functions::Tanh     cell_output_act_fn_t;

    struct ComputeGateFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;

        const char   *patTypes;

        const real_t *ugBiasWeights;
        const real_t *rgBiasWeights;

        real_t *ugActs;
        real_t *rgActs;

        real_t *outputs;

        __host__ __device__ real_t operator() (const int &outputIdx, const thrust::tuple<bool, bool> &t) const
        {
            // unpack the tuple
            bool firstCall    = t.get<0>();
            bool checkPatType = t.get<1>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set the all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    if (prevOutputDistance > 0)
                        outputs[outputIdx] = 0;
                    return 0;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

            // load the niag activations
            real_t rgAct = rgActs[outputIdx];
            real_t ugAct = ugActs[outputIdx];

            // add bias activations
            rgAct += bias * rgBiasWeights[blockIdx];
            ugAct += bias * ugBiasWeights[blockIdx];

            // apply the activation functions
            rgAct = gate_act_fn_t::fn(rgAct);
            ugAct = gate_act_fn_t::fn(ugAct);

            // store the niag activations
            rgActs[outputIdx] = rgAct;
            ugActs[outputIdx] = ugAct;

            // return reset output
            if (firstCall)
                return 0;
            else
                return outputs[outputIdx + prevOutputDistance] * rgAct;
        }
    };

    struct ComputeBlockOutputFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;

        const char   *patTypes;
        real_t *outputs;

        const real_t *niBiasWeights;
        const real_t *ugBiasWeights;

        real_t *niActs;
        real_t *ugActs;

        __host__ __device__ real_t operator() (const int &outputIdx, const thrust::tuple<bool, bool> &t) const
        {
            // unpack the tuple
            bool firstCall    = t.get<0>();
            bool checkPatType = t.get<1>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set the all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    return 0;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

            // load the niag activations
            real_t niAct = niActs[outputIdx];

            // add bias activations
            niAct += bias * niBiasWeights[blockIdx];

            // apply the activation functions
            niAct = cell_input_act_fn_t::fn(niAct);

            // store the niag activations
            niActs[outputIdx] = niAct;

            // calculate the cell state and store the result
            real_t ugAct = ugActs[outputIdx];
            real_t output = niAct * ugAct;

            if (!firstCall)
                output += outputs[outputIdx + prevOutputDistance] * (1.0 - ugAct);

            return output;
        }
    };

    struct ResortOutputsFn
    {
        int layerSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return bwOutputs[offset - effLayerSize];
        }
    };

    struct ResortOutputErrorsFn
    {
        int layerSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
                bwOutputErrors[offset - effLayerSize] = outputErr;
        }
    };

    struct ComputeBlockErrorsFn
    {
        int effLayerSize;
        int prevOutputDistance;

        const char   *patTypes;
        const real_t *outputs;
        const real_t *rgOutputErrors;

        const real_t *niActs;
        const real_t *ugActs;
        const real_t *rgActs;

        real_t *niDeltas;
        real_t *ugDeltas;
        real_t *rgDeltas;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int, bool, bool, bool> &t) const
        {
            // unpack the tuple
            real_t outputErr    = t.get<0>();
            int    outputIdx    = t.get<1>();
            bool   firstCall    = t.get<2>();
            bool   lastCall     = t.get<3>();
            bool   checkPatType = t.get<4>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    niDeltas       [outputIdx] = 0;
                    ugDeltas       [outputIdx] = 0;
                    rgDeltas       [outputIdx] = 0;
                    return;
                }
            }

            // load the niag activations, the cell state and the output error
            real_t niAct     = niActs      [outputIdx];
            real_t ugAct     = ugActs      [outputIdx];
            real_t rgAct     = rgActs      [outputIdx];
            real_t cumuErr   = outputErr;

            if (!firstCall) {
                real_t nextUgDelta      = ugDeltas       [outputIdx - prevOutputDistance];
                real_t nextRgDelta      = rgDeltas       [outputIdx - prevOutputDistance];

                cumuErr += nextUgDelta + nextRgDelta;
            }

            // calculate the net input delta
            real_t niDelta = cell_input_act_fn_t::deriv(niAct) * ugAct * cumuErr;

            // calculate the net update and reset delta
            real_t ugDelta = niAct;
            real_t rgDelta = 0;

            if (!lastCall) {
                real_t rgOutputErr = rgOutputErrors[outputIdx];
                real_t prevOutput  = outputs       [outputIdx + prevOutputDistance];

                ugDelta -= prevOutput;
                rgDelta  = gate_act_fn_t::deriv(rgAct) * prevOutput * rgOutputErr;
            }
            ugDelta *= gate_act_fn_t::deriv(ugAct) * cumuErr;

            // store the niag deltas and the cell state error
            niDeltas       [outputIdx] = helpers::limitedError(niDelta);
            ugDeltas       [outputIdx] = helpers::limitedError(ugDelta);
            rgDeltas       [outputIdx] = helpers::limitedError(rgDelta);
        }
    };

    struct ComputeWeightUpdateFn
    {
        int    layerSize;
        int    effLayerSize;
        int    precLayerSize;
        int    timestepDistance;
        int    parallelSequences;
        int    patternsCount;
        int    biasWeightsOffset;
        int    internalWeightsOffset;
        real_t bias;

        const real_t *plOutputs;
        const real_t *fwOutputs;   
        const real_t *bwOutputs;   
        const real_t *fwNiDeltas;  
        const real_t *bwNiDeltas;  
        const real_t *fwUgDeltas;  
        const real_t *bwUgDeltas;  
        const real_t *fwRgDeltas;  
        const real_t *bwRgDeltas;  

        __host__ __device__ real_t operator() (const int &weightIdx) const
        {
            // determine the weight type
            // 
            // weightType = 0bXXYY with XX = {input, bias, internal}
            //                     and  YY = {NI, UG, RG}
            //
            // weightType = 0b0000 ( 0): NI input weight
            //              0b0001 ( 1): UG input weight
            //              0b0010 ( 2): RG input weight
            //              0b0011 ( 3): not used
            //              0b0100 ( 4): NI bias weight
            //              0b0101 ( 5): UG bias weight
            //              0b0110 ( 6): RG bias weight
            //              0b0111 ( 7): not used
            //              0b1000 ( 8): NI internal weight
            //              0b1010 ( 9): UG internal weight
            //              0b1011 (10): RG internal weight
            //              0b1001 (11): not used
            //              0b1100 (12): not used
            int inwc = layerSize * precLayerSize;
            int biwc = layerSize;
            int itwc = layerSize * effLayerSize;

            int weightType = (int)(weightIdx >= 0                     + 1 * inwc) +
                             (int)(weightIdx >= 0                     + 2 * inwc) +
                             (int)(weightIdx >= 0                     + 3 * inwc) * 2 +
                             (int)(weightIdx >= biasWeightsOffset     + 1 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 2 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 3 * biwc) * 2 +
                             (int)(weightIdx >= internalWeightsOffset + 1 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 2 * itwc);

            int weightTypeX = weightType & 0xC;
            int weightTypeY = weightType & 0x3;

            // calculate indices, offsets and increments 
            const real_t *offOutputs;
            int           tgtBlockIdx;
            int           offOutputsInc;
            bool          skipFirstPattern = false;
            bool          skipLastPattern  = false;
            bool          isBwStateWeight;

            switch (weightTypeX) {
            // input weight
            case 0x0: 
                {{
                    // calculate indices
                    int inputWeightIdx = weightIdx;
                    int plBlockIdx     = inputWeightIdx % precLayerSize;
                    int blockIdx       = (inputWeightIdx - weightTypeY * (biasWeightsOffset/3)) / precLayerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &plOutputs[plBlockIdx];
                    offOutputsInc = precLayerSize;
                }}
                break;
            // bias weight
            case 0x4: 
                {{
                    // calculate indices
                    int biasWeightIdx = weightIdx - biasWeightsOffset;
                    int blockIdx      = biasWeightIdx - weightTypeY * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = NULL;
                    offOutputsInc = 0;
                }}
                break;

            // internal weight
            default: 
                {{
                    // calculate indices
                    int internalWeightIdx = weightIdx - internalWeightsOffset;
                    int srcBlockIdx       = internalWeightIdx % effLayerSize;
                    int blockIdx          = internalWeightIdx / effLayerSize - weightTypeY * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = (isBwStateWeight ? &bwOutputs[srcBlockIdx] : &fwOutputs[srcBlockIdx]);
                    offOutputsInc = effLayerSize;

                    if (isBwStateWeight) {
                        offOutputs += timestepDistance;
                        skipLastPattern = true;
                    }
                    else {
                        offOutputs -= timestepDistance;
                        skipFirstPattern = true;
                    }
                }}
                break;
            }

            // determine the start of the delta values
            const real_t *niagDeltasLut[] = {
                fwNiDeltas,
                fwUgDeltas,
                fwRgDeltas,
                bwNiDeltas,
                bwUgDeltas,
                bwRgDeltas
            };

            // calculate the weight update over all patterns            
            const real_t *offDeltas = &niagDeltasLut[weightTypeY + (isBwStateWeight ? 3 : 0)][tgtBlockIdx];

            if (skipFirstPattern) {
                offOutputs += parallelSequences * offOutputsInc;
                offDeltas  += parallelSequences * effLayerSize;
            }

            int numPatterns = patternsCount;
            if (skipFirstPattern || skipLastPattern)
                numPatterns -= parallelSequences;

            real_t wu = 0;
            for (int i = 0; i < numPatterns; ++i) {
                wu += (offOutputs ? *offOutputs : bias) * *offDeltas;

                offOutputs += offOutputsInc;
                offDeltas  += effLayerSize;
            }

            return wu;
        }
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice>
    GruLayer<TDevice>::GruLayer(const helpers::JsonValue &layerChild, 
                                  const helpers::JsonValue &weightsSection,
                                  Layer<TDevice> &precedingLayer,
                                  bool bidirectional)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 3, (bidirectional ? (3 * helpers::safeJsonGetInt(layerChild, "size")) >> 1 : (3 * helpers::safeJsonGetInt(layerChild, "size"))), precedingLayer)
        , m_isBidirectional      (bidirectional)
    {
        if (m_isBidirectional && this->size() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with an odd layer size");

        // set raw pointers
        int ls  = this->size();
        int pls = this->precedingLayer().size();

        _rawNiBiasWeights     = helpers::getRawPointer(this->weights()) + 3 * ls * pls + 0 * ls;
        _rawUgBiasWeights     = helpers::getRawPointer(this->weights()) + 3 * ls * pls + 1 * ls;
        _rawRgBiasWeights     = helpers::getRawPointer(this->weights()) + 3 * ls * pls + 2 * ls;

        // create the forward and backward info structs
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            // calculate sizes
            const int pls = this->precedingLayer().size();
            const int ls  = this->size();
            const int els = this->size() / (m_isBidirectional ? 2 : 1);
            const int rows   = this->size() / (m_isBidirectional ? 2 : 1);
            const int cols   = this->parallelSequences();

            // cell states, niags, deltas, ...
            Cpu::real_vector tmp(this->outputs().size() / (m_isBidirectional ? 2 : 1), 0);

            if (m_isBidirectional) {
                fwbw->tmpOutputs      = tmp;
                fwbw->tmpOutputErrors = tmp;
            }
            else {
                fwbw->tmpOutputs     .swap(this->_outputs());
                fwbw->tmpOutputErrors.swap(this->outputErrors());
            }

            fwbw->niActs       = tmp;
            fwbw->ugActs       = tmp;
            fwbw->rgActs       = tmp;
            fwbw->niDeltas     = tmp;
            fwbw->ugDeltas     = tmp;
            fwbw->rgDeltas     = tmp;
            fwbw->tmpRgOutputs = tmp;

            // weight matrices
            weight_matrices_t* wmArr [] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
            real_vector*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
            for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
                weight_matrices_t *wm  = wmArr [wmArrIdx];
                real_vector       *wts = wtsArr[wmArrIdx];

                int numInputWeights      = ls * pls;
                int numInternalWeights   = ls * els;
                int inputWeightsStart    = ((fwbwArrIdx == 1) ? (numInputWeights    / 2) : 0);
                int internalWeightsStart = ((fwbwArrIdx == 1) ? (numInternalWeights / 2) : 0) + 3 * (ls * (pls + 1));

                wm->niInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 0 * numInputWeights);
                wm->ugInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 1 * numInputWeights);
                wm->rgInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 2 * numInputWeights);

                wm->niInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 0 * numInternalWeights);
                wm->ugInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 1 * numInternalWeights);
                wm->rgInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 2 * numInternalWeights);
            }

            // matrices for each timestep
            for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int offset = timestep * rows * cols;

                timestep_matrices_t tm;
                tm.tmpOutputs      = helpers::Matrix<TDevice>(&fwbw->tmpOutputs,      rows, cols, offset);
                tm.tmpOutputErrors = helpers::Matrix<TDevice>(&fwbw->tmpOutputErrors, rows, cols, offset);
                tm.tmpRgOutputs    = helpers::Matrix<TDevice>(&fwbw->tmpRgOutputs,    rows, cols, offset);
                tm.niActs          = helpers::Matrix<TDevice>(&fwbw->niActs,          rows, cols, offset);
                tm.ugActs          = helpers::Matrix<TDevice>(&fwbw->ugActs,          rows, cols, offset);
                tm.rgActs          = helpers::Matrix<TDevice>(&fwbw->rgActs,          rows, cols, offset);
                tm.niDeltas        = helpers::Matrix<TDevice>(&fwbw->niDeltas,        rows, cols, offset);
                tm.ugDeltas        = helpers::Matrix<TDevice>(&fwbw->ugDeltas,        rows, cols, offset);
                tm.rgDeltas        = helpers::Matrix<TDevice>(&fwbw->rgDeltas,        rows, cols, offset);

                fwbw->timestepMatrices.push_back(tm);
            }
        }

        if (!m_isBidirectional) {
            m_fw.tmpOutputs     .swap(this->_outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }
    }

    template <typename TDevice>
    GruLayer<TDevice>::~GruLayer()
    {
    }

    template <typename TDevice>
    const std::string& GruLayer<TDevice>::type() const
    {
        static const std::string su("gru");
        static const std::string sb("bgru");
        return (m_isBidirectional ? sb : su);
    }

    template <typename TDevice>
    bool GruLayer<TDevice>::isBidirectional() const
    {
        return m_isBidirectional;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& GruLayer<TDevice>::netInputActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& GruLayer<TDevice>::netInputDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& GruLayer<TDevice>::updateGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ugActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& GruLayer<TDevice>::updateGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ugDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& GruLayer<TDevice>::resetGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.rgActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& GruLayer<TDevice>::resetGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.rgDeltas;
    }

    template <typename TDevice>
    void GruLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        TrainableLayer<TDevice>::loadSequences(fraction);

        m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());

        // update the niag matrices
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            int rows = this->size() / (m_isBidirectional ? 2 : 1);
            int cols = this->curMaxSeqLength() * this->parallelSequences();

            fwbw->niActsMatrix = helpers::Matrix<TDevice>(&fwbw->niActs, rows, cols);
            fwbw->ugActsMatrix = helpers::Matrix<TDevice>(&fwbw->ugActs, rows, cols);
            fwbw->rgActsMatrix = helpers::Matrix<TDevice>(&fwbw->rgActs, rows, cols);

            fwbw->niDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->niDeltas, rows, cols);
            fwbw->ugDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->ugDeltas, rows, cols);
            fwbw->rgDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->rgDeltas, rows, cols);
        }
    }

    template <typename TDevice>
    void GruLayer<TDevice>::computeForwardPass()
    {
        // for unidirectional GRU, we can write the outputs directly in the layer output vector
        if (!m_isBidirectional) {
            m_fw.tmpOutputs.swap(this->_outputs());
        }

        // sum up the activations from the preceding layer
        {{
            // forward states
            m_fw.niActsMatrix.assignProduct(m_fw.weightMatrices.niInput, true, m_precLayerOutputsMatrix, false);
            m_fw.ugActsMatrix.assignProduct(m_fw.weightMatrices.ugInput, true, m_precLayerOutputsMatrix, false);
            m_fw.rgActsMatrix.assignProduct(m_fw.weightMatrices.rgInput, true, m_precLayerOutputsMatrix, false);

            // backward states
            if (m_isBidirectional) {
                m_bw.niActsMatrix.assignProduct(m_bw.weightMatrices.niInput, true, m_precLayerOutputsMatrix, false);
                m_bw.ugActsMatrix.assignProduct(m_bw.weightMatrices.ugInput, true, m_precLayerOutputsMatrix, false);
                m_bw.rgActsMatrix.assignProduct(m_bw.weightMatrices.rgInput, true, m_precLayerOutputsMatrix, false);
            }
        }}

        // compute the block outputs
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeGateFn       gfn;
            internal::ComputeBlockOutputFn fn;
            gfn.effLayerSize       = fn.effLayerSize       = els;
            gfn.prevOutputDistance = fn.prevOutputDistance = -n;
            gfn.bias               = fn.bias               = this->bias();
            gfn.patTypes           = fn.patTypes           = helpers::getRawPointer(this->patTypes());
            gfn.outputs            = fn.outputs            = helpers::getRawPointer(m_fw.tmpOutputs);
            gfn.ugActs             = fn.ugActs             = helpers::getRawPointer(m_fw.ugActs);
            gfn.ugBiasWeights      = fn.ugBiasWeights      = _rawUgBiasWeights;
            gfn.rgActs             = helpers::getRawPointer(m_fw.rgActs);;
            gfn.rgBiasWeights      = _rawRgBiasWeights;
            fn.niActs              = helpers::getRawPointer(m_fw.niActs);
            fn.niBiasWeights       = _rawNiBiasWeights;

            for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                // collect outputs from previous timestep
                if (timestep != 0) {
                    m_fw.timestepMatrices[timestep].ugActs.addProduct(m_fw.weightMatrices.ugInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                    m_fw.timestepMatrices[timestep].rgActs.addProduct(m_fw.weightMatrices.rgInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                }

                // compute gates
                thrust::transform(
                    thrust::counting_iterator<int>(n*timestep),
                    thrust::counting_iterator<int>(n*timestep) + n,
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            thrust::constant_iterator<bool>(!timestep),
                            thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    m_fw.tmpRgOutputs.begin() + n*timestep,
                    gfn
                    );

                // collect outputs from previous timestep
                if (timestep != 0) {
                    m_fw.timestepMatrices[timestep].niActs.addProduct(m_fw.weightMatrices.niInternal, true, m_fw.timestepMatrices[timestep-1].tmpRgOutputs, false);
                }

                // compute outputs
                thrust::transform(
                    thrust::counting_iterator<int>(n*timestep),
                    thrust::counting_iterator<int>(n*timestep) + n,
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            thrust::constant_iterator<bool>(!timestep),
                            thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    m_fw.tmpOutputs.begin() + n*timestep,
                    fn
                    );
            }

            // backward states
            if (m_isBidirectional) {
                gfn.prevOutputDistance = fn.prevOutputDistance = +n;
                gfn.ugBiasWeights     += els;
                gfn.rgBiasWeights     += els;
                fn.niBiasWeights      += els;
                fn.ugBiasWeights      += els;
                gfn.outputs            = fn.outputs = helpers::getRawPointer(m_bw.tmpOutputs);
                gfn.ugActs             = fn.ugActs  = helpers::getRawPointer(m_bw.ugActs);
                fn.niActs              = helpers::getRawPointer(m_bw.niActs);
                gfn.rgActs             = helpers::getRawPointer(m_bw.ugActs);

                for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                    // collect outputs from previous timestep
                    if (timestep != this->curMaxSeqLength()-1) {
                        m_bw.timestepMatrices[timestep].ugActs.addProduct(m_bw.weightMatrices.ugInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].rgActs.addProduct(m_bw.weightMatrices.rgInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                    }

                    // compute outputs
                    thrust::transform(
                        thrust::counting_iterator<int>(n*timestep),
                        thrust::counting_iterator<int>(n*timestep) + n,
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1), thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                        m_bw.tmpRgOutputs.begin() + n*timestep,
                        gfn
                        );

                    // collect outputs from previous timestep
                    if (timestep != this->curMaxSeqLength()-1) {
                        m_bw.timestepMatrices[timestep].niActs.addProduct(m_bw.weightMatrices.niInternal, true, m_bw.timestepMatrices[timestep+1].tmpRgOutputs, false);
                    }

                    // compute outputs
                    thrust::transform(
                        thrust::counting_iterator<int>(n*timestep),
                        thrust::counting_iterator<int>(n*timestep) + n,
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1), thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                        m_bw.tmpOutputs.begin() + n*timestep,
                        fn
                        );
                }
            }
        }}

        // resort outputs
        if (m_isBidirectional) {
            internal::ResortOutputsFn fn;
            fn.layerSize    = this->size();
            fn.effLayerSize = this->size() / 2;
            fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs    = helpers::getRawPointer(m_bw.tmpOutputs);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences() * this->size(),
                this->_outputs().begin(),
                fn
                );
        }
        else {
            this->_outputs().swap(m_fw.tmpOutputs);
        }
    }

    template <typename TDevice>
    void GruLayer<TDevice>::computeBackwardPass()
    {
        // for unidirectional GRU, we can write the output errors directly in the layer output errors vector
        if (m_isBidirectional) {
            internal::ResortOutputErrorsFn fn;
            fn.layerSize      = this->size();
            fn.effLayerSize   = this->size() / 2;
            fn.fwOutputErrors = helpers::getRawPointer(m_fw.tmpOutputErrors);
            fn.bwOutputErrors = helpers::getRawPointer(m_bw.tmpOutputErrors);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }
        else {
            m_fw.tmpOutputs     .swap(this->outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }

        // calculate the block errors
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockErrorsFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.niActs             = helpers::getRawPointer(m_fw.niActs);
            fn.ugActs             = helpers::getRawPointer(m_fw.ugActs);
            fn.rgActs             = helpers::getRawPointer(m_fw.rgActs);
            fn.niDeltas           = helpers::getRawPointer(m_fw.niDeltas);
            fn.ugDeltas           = helpers::getRawPointer(m_fw.ugDeltas);
            fn.rgDeltas           = helpers::getRawPointer(m_fw.rgDeltas);
            fn.outputs            = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.rgOutputErrors     = helpers::getRawPointer(m_fw.tmpRgOutputs);

            for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                // collect errors from previous timestep
                if (timestep != this->curMaxSeqLength()-1) {
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.niInternal, false, m_fw.timestepMatrices[timestep+1].niDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.ugInternal, false, m_fw.timestepMatrices[timestep+1].ugDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpRgOutputs.assignProduct(m_fw.weightMatrices.niInternal, false, m_fw.timestepMatrices[timestep+1].niDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.rgInternal, false, m_fw.timestepMatrices[timestep+1].tmpRgOutputs, false);
                }

                // compute errors
                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(m_fw.tmpOutputErrors.begin() + n*timestep,   thrust::counting_iterator<int>(n*timestep),   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),   thrust::constant_iterator<bool>(!timestep),   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    thrust::make_zip_iterator(thrust::make_tuple(m_fw.tmpOutputErrors.begin() + n*timestep+n, thrust::counting_iterator<int>(n*timestep)+n, thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n, thrust::constant_iterator<bool>(!timestep)+n, thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                    fn
                    );
            }

            // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.niActs             = helpers::getRawPointer(m_bw.niActs);
                fn.ugActs             = helpers::getRawPointer(m_bw.ugActs);
                fn.rgActs             = helpers::getRawPointer(m_bw.rgActs);
                fn.niDeltas           = helpers::getRawPointer(m_bw.niDeltas);
                fn.ugDeltas           = helpers::getRawPointer(m_bw.ugDeltas);
                fn.rgDeltas           = helpers::getRawPointer(m_bw.rgDeltas);
                fn.outputs            = helpers::getRawPointer(m_bw.tmpOutputs);
                fn.rgOutputErrors     = helpers::getRawPointer(m_bw.tmpRgOutputs);

                for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                    // collect errors from previous timestep
                    if (timestep != 0) {
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.niInternal, false, m_bw.timestepMatrices[timestep-1].niDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.ugInternal, false, m_bw.timestepMatrices[timestep-1].ugDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpRgOutputs.assignProduct(m_bw.weightMatrices.niInternal, false, m_bw.timestepMatrices[timestep-1].niDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.rgInternal, false, m_bw.timestepMatrices[timestep-1].tmpRgOutputs, false);
                    }

                    // compute errors
                    thrust::for_each(
                        thrust::make_zip_iterator(thrust::make_tuple(m_bw.tmpOutputErrors.begin() + n*timestep,   thrust::counting_iterator<int>(n*timestep),   thrust::constant_iterator<bool>(!timestep),   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                        thrust::make_zip_iterator(thrust::make_tuple(m_bw.tmpOutputErrors.begin() + n*timestep+n, thrust::counting_iterator<int>(n*timestep)+n, thrust::constant_iterator<bool>(!timestep)+n, thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n, thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                        fn
                        );
                }
            }
        }}

        // back-propagate the error to the preceding layer
        {{
            TrainableLayer<TDevice> *pl = dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
            if (pl) {
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(), pl->size(), this->curMaxSeqLength() * this->parallelSequences());

                // forward states
                plErrorsMatrix.assignProduct(m_fw.weightMatrices.niInput, false, m_fw.niDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.ugInput, false, m_fw.ugDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.rgInput, false, m_fw.rgDeltasMatrix, false);

                // backward states
                if (m_isBidirectional) {
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.niInput, false, m_bw.niDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.ugInput, false, m_bw.ugDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.rgInput, false, m_bw.rgDeltasMatrix, false);
                }
            }
        }}

        // compute the weight updates
        {{
            internal::ComputeWeightUpdateFn fn;
            fn.layerSize             = this->size();
            fn.effLayerSize          = this->size() / (m_isBidirectional ? 2 : 1);
            fn.precLayerSize         = this->precedingLayer().size();
            fn.timestepDistance      = this->parallelSequences() * this->size() / (m_isBidirectional ? 2 : 1);
            fn.parallelSequences     = this->parallelSequences();
            fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
            fn.biasWeightsOffset     = this->size() * this->precedingLayer().size() * 3;
            fn.internalWeightsOffset = fn.biasWeightsOffset + this->size() * 3;
            fn.bias                  = this->bias();
            fn.plOutputs             = helpers::getRawPointer(this->precedingLayer().outputs());
            fn.fwOutputs             = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs             = helpers::getRawPointer(m_bw.tmpOutputs);
            fn.fwNiDeltas            = helpers::getRawPointer(m_fw.niDeltas);
            fn.bwNiDeltas            = helpers::getRawPointer(m_bw.niDeltas);
            fn.fwUgDeltas            = helpers::getRawPointer(m_fw.ugDeltas);
            fn.bwUgDeltas            = helpers::getRawPointer(m_bw.ugDeltas);
            fn.fwRgDeltas            = helpers::getRawPointer(m_fw.rgDeltas);
            fn.bwRgDeltas            = helpers::getRawPointer(m_bw.rgDeltas);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + (int)this->weightUpdates().size(),
                this->_weightUpdates().begin(),
                fn
                );
        }}

        // re-swap the output errors and the tmp output errors of the forward pass
        if (!m_isBidirectional) {
            this->outputErrors().swap(m_fw.tmpOutputErrors);
            this->_outputs()    .swap(m_fw.tmpOutputs);
        }
    }


    // explicit template instantiations
    template class GruLayer<Cpu>;
    template class GruLayer<Gpu>;

} // namespace layers
