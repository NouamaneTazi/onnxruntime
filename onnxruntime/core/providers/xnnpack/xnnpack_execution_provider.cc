// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <unordered_set>

#include "xnnpack_execution_provider.h"
#include "detail/utils.h"
#include "detail/node_support_checker.h"

#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/shared/node_unit/node_unit.h"

#include <xnnpack.h>

namespace onnxruntime {

namespace xnnpack {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<                             \
      ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op) \
  BuildKernelCreateInfo<              \
      ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, Start, Op)>

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 11, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 10, uint8_t, QLinearConv);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 11, 11, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 12, MaxPool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSInternalNHWCDomain, 7, 7, AveragePool);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSDomain, 1, uint8_t, QLinearAveragePool);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 1, 12, Softmax);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSDomain, 1, 12, QLinearSoftmax);

std::unique_ptr<KernelRegistry> RegisterKernels() {
  auto kernel_registry = std::make_unique<onnxruntime::KernelRegistry>();

  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
      KERNEL_CREATE_INFO(11, Conv),
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider,
                                                                  kMSInternalNHWCDomain, 10, uint8_t, QLinearConv)>,
      KERNEL_CREATE_INFO_VERSIONED(11, 11, MaxPool),
      KERNEL_CREATE_INFO(12, MaxPool),
      KERNEL_CREATE_INFO_VERSIONED(7, 7, AveragePool),
      BuildKernelCreateInfo<
          ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSDomain, 1, uint8_t, QLinearAveragePool)>,
      BuildKernelCreateInfo<
          ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kOnnxDomain, 1, 12, Softmax)>,
      BuildKernelCreateInfo<
          ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kXnnpackExecutionProvider, kMSDomain, 1, 12, QLinearSoftmax)>,

  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  return kernel_registry;
}

}  // namespace xnnpack

using namespace xnnpack;

XnnpackExecutionProvider::XnnpackExecutionProvider(const XnnpackExecutionProviderInfo& /*info*/)
    : IExecutionProvider{kXnnpackExecutionProvider, true} {
}

// implement RegisterAllocator to test/validate sharing the CPU EP's allocator
void XnnpackExecutionProvider::RegisterAllocator(AllocatorManager& allocator_manager) {
  OrtDevice cpu_device{OrtDevice::CPU, OrtDevice::MemType::DEFAULT, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};

  // if EP is used in multiple inference sessions we may already have an allocator. if so use that.
  auto cpu_alloc = GetAllocator(cpu_device.Id(), OrtMemTypeDefault);
  if (!cpu_alloc) {
    // use shared allocator if available
    cpu_alloc = allocator_manager.GetAllocator(OrtMemTypeDefault, cpu_device);

    if (!cpu_alloc) {
      // create our allocator
      AllocatorCreationInfo allocator_info(
          [](int) {
            return std::make_unique<CPUAllocator>(OrtMemoryInfo(kXnnpackExecutionProvider,
                                                                OrtAllocatorType::OrtDeviceAllocator));
          });

      cpu_alloc = CreateAllocator(allocator_info);
      // enable sharing of our allocator
      allocator_manager.InsertAllocator(cpu_alloc);
    }

    InsertAllocator(cpu_alloc);
  }

  // TODO: Create `struct xnn_allocator` that wraps cpu_allocator, and provide in the call to xnn_initialize so that
  //       xnnpack is using the ORT allocator.
  xnn_status st = xnn_initialize(nullptr);
  if (st != xnn_status_success) {
    ORT_THROW("XNNPACK initialization failed with status ", st);
  }
}

std::vector<std::unique_ptr<ComputeCapability>> XnnpackExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;

  std::shared_ptr<KernelRegistry> registry = GetKernelRegistry();
  std::unordered_set<const Node*> supported_nodes;
  NodeSupportChecker checker{graph, supported_nodes};

  std::unordered_map<const Node*, ComputeCapability*> node_to_compute_capability;

  // Get all the NodeUnits in the GraphViewer so we can check if something is in a QDQ node group
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = GetAllNodeUnits(graph);

  // This holds the result of whether a NodeUnit is supported or not,
  // to prevent nodes in a NodeUnit to be checked for multiple times
  std::unordered_map<const NodeUnit*, bool> node_unit_supported_result;
  node_unit_supported_result.reserve(node_unit_holder.size());
  std::unordered_set<NodeIndex> nodeIndex_map;
  for (NodeIndex idx : graph.GetNodesInTopologicalOrder()) {
    const Node* n = graph.GetNode(idx);
    if (n == nullptr) {
      continue;
    }
    const NodeUnit& node_unit = *node_unit_map[n];
    // if node is part of a QDQ group, we will mark it compatible in the first call as long as we support the target node.
    const Node& node = *n;
    bool request_node = false;
    // any node in NodeUnit will trigger IsNodeSupported, so we just check once.
    if (node_unit_supported_result.count(&node_unit)) {
      continue;
    } else if (node.GetExecutionProviderType() == "") {
      // unassigned node.
      // check if this is an ONNX operator that we have an NHWC xnnpack kernel for.
      if (checker.IsNodeSupported(node_unit)) {
        request_node = true;
        // we need to support quantizedOp fusion
      } else if (node_unit.UnitType() != NodeUnit::Type::QDQGroup) {
        // see if it's an activation we can fuse with a node we support. note that we can only do this after
        // the layout transform as we need to fuse with the NWHC op that we have the real kernel for.
        const Node* fuse_with = checker.IsNodeSupportedWithFusion(node);
        if (fuse_with) {
          // add new MetaDef to existing ComputeCapability.
          // we know an entry must exist in node_to_compute_capability as we update supported_nodes
          // when creating the ComputeCapability, and the logic in IsNodeSupportedWithFusion
          // checks the fuse_with node is in supported_nodes.
          auto iter = node_to_compute_capability.find(fuse_with);
          ORT_ENFORCE(iter != node_to_compute_capability.cend(),
                      "node_to_compute_capability is not in sync with supported_nodes.");

          // update the MetaDef to cover the nodes being fused.
          // the fused node will have OpType:'Conv' and Domain:kMSInternalNHWCDomain.
          // GraphPartitioner will match the statically registered xnnpack NHWC Conv kernel instead of
          // calling IExecutionProvider::Compile
          ComputeCapability& capability = *iter->second;
          capability.sub_graph->SetMetaDef(FuseActivation(*fuse_with, node, graph));
          capability.sub_graph->nodes.push_back(node.Index());
          capability.sub_graph->use_existing_schema = true;
        }
      }
    } else if (node.GetExecutionProviderType() == Type()) {
      // second call to GetCapability after layout changes.
      // as we requested the node in the first call, it should be supported in the second call.
      request_node = true;
      // we will create a quantized Op to replace QDQGroup in the second call, then we can fuse it with activation after
      if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
        // create a ComputeCapability for the QDQGroup node.
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        for (const auto& node_i : node_unit.GetDQNodes()) {
          sub_graph->nodes.push_back(node_i->Index());
        }
        sub_graph->nodes.push_back(node_unit.Index());
        for (const auto& node_o : node_unit.GetQNodes()) {
          sub_graph->nodes.push_back(node_o->Index());
        }
        sub_graph->SetMetaDef(std::move(FuseQDQGroup(node_unit)));
        sub_graph->use_existing_schema = true;
        capabilities.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
        node_to_compute_capability.insert({&node_unit.GetNode(), capabilities.back().get()});

        supported_nodes.insert(&node_unit.GetNode());
        node_unit_supported_result[&node_unit] = request_node;
        continue;
      }
    } else {
      // node belongs to another EP
      continue;
    }
    node_unit_supported_result[&node_unit] = request_node;
    if (request_node) {
      // for a QDQgroup, we will request its DQ/target/Q nodes.
      // for SingleNode single node, its DQ/Q will be empty
      auto request_single_node = [&](const Node& inode) {
        // create a ComputeCapability for the individual node.
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(inode.Index());
        capabilities.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));

        node_to_compute_capability.insert({&node, capabilities.back().get()});
        supported_nodes.insert(&node);
      };
      // T. we assume all the inputs are either DQnodes or constant initializer.
      for (auto dq_node : node_unit.GetDQNodes()) {
        request_single_node(*dq_node);
      }
      request_single_node(node_unit.GetNode());
      for (auto q_node : node_unit.GetQNodes()) {
        if (q_node->Index() != node_unit.Index()) {
          request_single_node(*q_node);
        }
      }
    }
  }

  // FUTURE: finding nodes to compile can be inserted here and added to the ComputeCapability instances returned.
  // GraphPartitioner can handle a mix of static and compiled kernels.

  return capabilities;
}

std::shared_ptr<KernelRegistry> XnnpackExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> registry = xnnpack::RegisterKernels();
  return registry;
}

XnnpackExecutionProvider::~XnnpackExecutionProvider() {
  xnn_deinitialize();
}

}  // namespace onnxruntime
