/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hardshrinkPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;

namespace
{
const char* HARDSHRINK_PLUGIN_VERSION{"1"};
const char* HARDSHRINK_PLUGIN_NAME{"Hardshrink"};
} // namespace

PluginFieldCollection HardshrinkPluginCreator::mFC{};
std::vector<PluginField> HardshrinkPluginCreator::mPluginAttributes;

HardshrinkPlugin::HardshrinkPlugin() {}

HardshrinkPlugin::HardshrinkPlugin(float lambd)
    : mLambd(lambd)
    , mBatchDim(1)
{
}

HardshrinkPlugin::HardshrinkPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mLambd = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int HardshrinkPlugin::getNbOutputs() const
{
    return 1;
}

int HardshrinkPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void HardshrinkPlugin::terminate() {}

Dims HardshrinkPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // Dimentions are untouched
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = inputs->nbDims;
    dimsOutput.d[0] = inputs->d[0];
    dimsOutput.d[1] = inputs->d[1];
    dimsOutput.d[2] = inputs->d[2];
    dimsOutput.d[3] = inputs->d[3];
    return dimsOutput;
}

size_t HardshrinkPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t HardshrinkPlugin::getSerializationSize() const
{
    // mLambd, mBatchDim
    return sizeof(float) + sizeof(int);
}

void HardshrinkPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mLambd);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void HardshrinkPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(mBatchDim == 1);
    ASSERT(nbOutputs == 1);
    // Collect in-batch size to use later in enqueue
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

bool HardshrinkPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
}

const char* HardshrinkPlugin::getPluginType() const
{
    return HARDSHRINK_PLUGIN_NAME;
}

const char* HardshrinkPlugin::getPluginVersion() const
{
    return HARDSHRINK_PLUGIN_VERSION;
}

void HardshrinkPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* HardshrinkPlugin::clone() const
{
    auto* plugin = new HardshrinkPlugin(mLambd);
    return plugin;
}

void HardshrinkPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* HardshrinkPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType HardshrinkPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool HardshrinkPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool HardshrinkPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
HardshrinkPluginCreator::HardshrinkPluginCreator() {}

const char* HardshrinkPluginCreator::getPluginName() const
{
    return HARDSHRINK_PLUGIN_NAME;
}

const char* HardshrinkPluginCreator::getPluginVersion() const
{
    return HARDSHRINK_PLUGIN_VERSION;
}

const PluginFieldCollection* HardshrinkPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* HardshrinkPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    HardshrinkPlugin* plugin = new HardshrinkPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* HardshrinkPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    HardshrinkPlugin* plugin = new HardshrinkPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
