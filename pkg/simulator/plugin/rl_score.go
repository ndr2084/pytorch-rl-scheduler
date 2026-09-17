package plugin

import (
	"context"
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"

	"google.golang.org/grpc"

	rlpb "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/rlscore/proto"
	simontype "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/type"
	"github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/utils"
)

const rlPreScoreStateKey = "PreScore-RLScorePlugin"

// rlNodeResult is what the RL policy returned for one candidate node.
type rlNodeResult struct {
	score int64
	gpuId string
}

// rlPreScoreState caches one ScoreBatch RPC's results for the current pod so
// Score() (called once per candidate node by the scheduler framework) reads
// from memory instead of making a network round trip per node.
type rlPreScoreState struct {
	scores map[string]rlNodeResult // nodeName -> result
}

func (s *rlPreScoreState) Clone() framework.StateData { return s }

// RLScorePlugin scores node candidates by delegating to a policy served
// behind a gRPC sidecar (see pkg/rlscore/proto). The plugin itself holds no
// scheduling logic beyond the RPC call and response bookkeeping.
type RLScorePlugin struct {
	cfg    *simontype.RLScorePluginCfg
	handle framework.Handle
	client rlpb.RLScorerClient
}

var _ framework.ScorePlugin = &RLScorePlugin{}
var _ framework.PreScorePlugin = &RLScorePlugin{}

func NewRLScorePlugin(configuration runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	var cfg *simontype.RLScorePluginCfg
	if err := frameworkruntime.DecodeInto(configuration, &cfg); err != nil {
		return nil, err
	}
	if cfg == nil || cfg.InferenceAddr == "" {
		return nil, fmt.Errorf("RLScore plugin requires pluginConfig.args.inferenceAddr (host:port of the RL inference server)")
	}
	if cfg.TimeoutMs <= 0 {
		cfg.TimeoutMs = 100
	}

	conn, err := grpc.Dial(cfg.InferenceAddr, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("failed to dial RL inference server(%s): %v", cfg.InferenceAddr, err)
	}

	plugin := &RLScorePlugin{
		cfg:    cfg,
		handle: handle,
		client: rlpb.NewRLScorerClient(conn),
	}
	allocateGpuIdFunc[plugin.Name()] = plugin.allocateGpuId
	return plugin, nil
}

func (plugin *RLScorePlugin) Name() string {
	return simontype.RLScorePluginName
}

func toNodeResourceState(nodeRes simontype.NodeResource) *rlpb.ResourceState {
	milliGpuLeft := make([]float64, len(nodeRes.MilliGpuLeftList))
	for i, v := range nodeRes.MilliGpuLeftList {
		milliGpuLeft[i] = float64(v)
	}
	return &rlpb.ResourceState{
		MilliCpuLeft:     float64(nodeRes.MilliCpuLeft),
		MilliCpuCapacity: float64(nodeRes.MilliCpuCapacity),
		MilliGpuLeftList: milliGpuLeft,
		GpuNumber:        int32(nodeRes.GpuNumber),
	}
}

func toPodResourceState(podRes simontype.PodResource) *rlpb.ResourceState {
	return &rlpb.ResourceState{
		MilliCpu:  float64(podRes.MilliCpu),
		MilliGpu:  float64(podRes.MilliGpu),
		GpuNumber: int32(podRes.GpuNumber),
	}
}

// PreScore gathers every accessible candidate node's resource state and
// scores them in a single ScoreBatch RPC, caching the response in CycleState.
func (plugin *RLScorePlugin) PreScore(ctx context.Context, state *framework.CycleState, pod *corev1.Pod, nodes []*corev1.Node) *framework.Status {
	podRes := utils.GetPodResource(pod)
	podKey := utils.GeneratePodKey(pod)

	req := &rlpb.BatchScoreRequest{}
	var nodeNames []string
	for _, node := range nodes {
		nodeResPtr := utils.GetNodeResourceViaHandleAndName(plugin.handle, node.Name)
		if nodeResPtr == nil || !utils.IsNodeAccessibleToPod(*nodeResPtr, podRes) {
			continue
		}
		req.Requests = append(req.Requests, &rlpb.ScoreRequest{
			PodKey:   podKey,
			NodeName: node.Name,
			Node:     toNodeResourceState(*nodeResPtr),
			Pod:      toPodResourceState(podRes),
		})
		nodeNames = append(nodeNames, node.Name)
	}

	result := &rlPreScoreState{scores: map[string]rlNodeResult{}}
	if len(req.Requests) > 0 {
		callCtx, cancel := context.WithTimeout(ctx, time.Duration(plugin.cfg.TimeoutMs)*time.Millisecond)
		defer cancel()
		resp, err := plugin.client.ScoreBatch(callCtx, req)
		if err != nil {
			log.Errorf("[RLScore] ScoreBatch RPC failed for pod(%s): %v\n", podKey, err)
			return framework.NewStatus(framework.Error, fmt.Sprintf("RLScore inference call failed: %v", err))
		}
		if len(resp.Responses) != len(req.Requests) {
			return framework.NewStatus(framework.Error, "RLScore inference server returned a mismatched number of responses")
		}
		for i, r := range resp.Responses {
			result.scores[nodeNames[i]] = rlNodeResult{score: r.Score, gpuId: r.GpuId}
		}
	}

	state.Write(rlPreScoreStateKey, result)
	return framework.NewStatus(framework.Success)
}

func (plugin *RLScorePlugin) Score(ctx context.Context, state *framework.CycleState, pod *corev1.Pod, nodeName string) (int64, *framework.Status) {
	c, err := state.Read(rlPreScoreStateKey)
	if err != nil {
		return framework.MinNodeScore, framework.AsStatus(fmt.Errorf("reading %q from cycleState: %w", rlPreScoreStateKey, err))
	}
	s, ok := c.(*rlPreScoreState)
	if !ok {
		return framework.MinNodeScore, framework.AsStatus(fmt.Errorf("cannot convert saved state to RLScorePlugin.rlPreScoreState"))
	}

	result, ok := s.scores[nodeName]
	if !ok {
		// PreScore should have scored every node the Filter phase let through;
		// getting here means node accessibility was re-evaluated differently
		// between PreScore and Score, which should not happen.
		return framework.MinNodeScore, framework.NewStatus(framework.Error, fmt.Sprintf("node(%s) was not scored by the RL policy in PreScore", nodeName))
	}
	if result.score < framework.MinNodeScore || result.score > framework.MaxNodeScore {
		return framework.MinNodeScore, framework.NewStatus(framework.Error, fmt.Sprintf("RL policy returned out-of-range score %d for node(%s)", result.score, nodeName))
	}
	return result.score, framework.NewStatus(framework.Success)
}

func (plugin *RLScorePlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// allocateGpuId is invoked by Open-Gpu-Share's Reserve step, once a node has
// won scheduling, to pick which physical/shared GPU to bind the pod to. It
// re-queries the policy for that single node rather than reusing the
// PreScore cache, since CycleState is not threaded through this call path.
func (plugin *RLScorePlugin) allocateGpuId(nodeRes simontype.NodeResource, podRes simontype.PodResource, _ simontype.GpuPluginCfg, _ *simontype.TargetPodList) (gpuId string) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(plugin.cfg.TimeoutMs)*time.Millisecond)
	defer cancel()

	resp, err := plugin.client.ScorePlacement(ctx, &rlpb.ScoreRequest{
		NodeName: nodeRes.NodeName,
		Node:     toNodeResourceState(nodeRes),
		Pod:      toPodResourceState(podRes),
	})
	if err != nil || resp.GpuId == "" {
		if err != nil {
			log.Errorf("[RLScore] allocateGpuId RPC failed for node(%s): %v\n", nodeRes.NodeName, err)
		}
		return simontype.AllocateExclusiveGpuId(nodeRes, podRes)
	}
	return resp.GpuId
}
