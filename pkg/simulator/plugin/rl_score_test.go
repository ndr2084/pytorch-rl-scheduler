package plugin

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"

	rlpb "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/rlscore/proto"
	simontype "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/type"
)

// fakeRLScorer is a minimal in-memory stand-in for a real policy sidecar,
// used to exercise the hand-written gRPC plumbing in
// pkg/rlscore/proto/rlscore_grpc.go end-to-end.
type fakeRLScorer struct {
	rlpb.RLScorerServer
	gpuIdToReturn string
}

func (f *fakeRLScorer) ScorePlacement(_ context.Context, in *rlpb.ScoreRequest) (*rlpb.ScoreResponse, error) {
	// Deterministic stand-in "policy": prefer nodes with more free CPU.
	return &rlpb.ScoreResponse{Score: int64(in.Node.MilliCpuLeft) % 101, GpuId: f.gpuIdToReturn}, nil
}

func (f *fakeRLScorer) ScoreBatch(_ context.Context, in *rlpb.BatchScoreRequest) (*rlpb.BatchScoreResponse, error) {
	resp := &rlpb.BatchScoreResponse{}
	for _, req := range in.Requests {
		s, err := f.ScorePlacement(context.Background(), req)
		if err != nil {
			return nil, err
		}
		resp.Responses = append(resp.Responses, s)
	}
	return resp, nil
}

// startFakeRLServer starts fakeRLScorer on an ephemeral local port and
// returns a dialed client plus a cleanup func.
func startFakeRLServer(t *testing.T, gpuIdToReturn string) (rlpb.RLScorerClient, func()) {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	assert.NoError(t, err)

	s := grpc.NewServer()
	rlpb.RegisterRLScorerServer(s, &fakeRLScorer{gpuIdToReturn: gpuIdToReturn})
	go s.Serve(lis)

	dialCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(dialCtx, lis.Addr().String(), grpc.WithInsecure(), grpc.WithBlock())
	assert.NoError(t, err)

	return rlpb.NewRLScorerClient(conn), func() {
		conn.Close()
		s.Stop()
	}
}

func TestRLScorePlugin_ClientServerRoundTrip(t *testing.T) {
	client, cleanup := startFakeRLServer(t, "2")
	defer cleanup()

	nodeRes := simontype.NodeResource{NodeName: "node-a", MilliCpuLeft: 3000, MilliGpuLeftList: []int64{1000, 1000}, GpuNumber: 2}
	podRes := simontype.PodResource{MilliCpu: 100, MilliGpu: 1000, GpuNumber: 1}

	resp, err := client.ScorePlacement(context.Background(), &rlpb.ScoreRequest{
		NodeName: nodeRes.NodeName,
		Node:     toNodeResourceState(nodeRes),
		Pod:      toPodResourceState(podRes),
	})
	assert.NoError(t, err)
	assert.Equal(t, int64(3000%101), resp.Score)
	assert.Equal(t, "2", resp.GpuId)

	batchResp, err := client.ScoreBatch(context.Background(), &rlpb.BatchScoreRequest{
		Requests: []*rlpb.ScoreRequest{
			{NodeName: "node-a", Node: toNodeResourceState(nodeRes), Pod: toPodResourceState(podRes)},
			{NodeName: "node-b", Node: toNodeResourceState(simontype.NodeResource{MilliCpuLeft: 500}), Pod: toPodResourceState(podRes)},
		},
	})
	assert.NoError(t, err)
	assert.Len(t, batchResp.Responses, 2)
	assert.Equal(t, int64(3000%101), batchResp.Responses[0].Score)
	assert.Equal(t, int64(500%101), batchResp.Responses[1].Score)
}

func TestRLScorePlugin_AllocateGpuId(t *testing.T) {
	client, cleanup := startFakeRLServer(t, "1")
	defer cleanup()

	plugin := &RLScorePlugin{cfg: &simontype.RLScorePluginCfg{TimeoutMs: 500}, client: client}
	nodeRes := simontype.NodeResource{NodeName: "node-a", MilliCpuLeft: 1000, MilliGpuLeftList: []int64{1000, 200}, GpuNumber: 2}
	podRes := simontype.PodResource{MilliCpu: 100, MilliGpu: 500, GpuNumber: 1}

	gpuId := plugin.allocateGpuId(nodeRes, podRes, simontype.GpuPluginCfg{}, nil)
	assert.Equal(t, "1", gpuId)
}

func TestRLScorePlugin_AllocateGpuId_FallsBackWhenPolicyGivesNoAnswer(t *testing.T) {
	client, cleanup := startFakeRLServer(t, "") // policy declines to pick a GPU
	defer cleanup()

	plugin := &RLScorePlugin{cfg: &simontype.RLScorePluginCfg{TimeoutMs: 500}, client: client}
	nodeRes := simontype.NodeResource{NodeName: "node-a", MilliCpuLeft: 1000, MilliGpuLeftList: []int64{1000, 1000}, GpuNumber: 2}
	podRes := simontype.PodResource{MilliCpu: 100, MilliGpu: 1000, GpuNumber: 1}

	gpuId := plugin.allocateGpuId(nodeRes, podRes, simontype.GpuPluginCfg{}, nil)
	assert.Equal(t, simontype.AllocateExclusiveGpuId(nodeRes, podRes), gpuId)
}

func TestRLScorePlugin_Score(t *testing.T) {
	plugin := &RLScorePlugin{cfg: &simontype.RLScorePluginCfg{TimeoutMs: 500}}

	state := framework.NewCycleState()
	state.Write(rlPreScoreStateKey, &rlPreScoreState{
		scores: map[string]rlNodeResult{
			"node-a": {score: 80, gpuId: "0"},
			"node-b": {score: 150, gpuId: "0"}, // out of range, should error
		},
	})

	score, status := plugin.Score(context.Background(), state, &corev1.Pod{}, "node-a")
	assert.True(t, status.IsSuccess())
	assert.Equal(t, int64(80), score)

	_, status = plugin.Score(context.Background(), state, &corev1.Pod{}, "node-b")
	assert.False(t, status.IsSuccess())

	_, status = plugin.Score(context.Background(), state, &corev1.Pod{}, "node-missing")
	assert.False(t, status.IsSuccess())
}
