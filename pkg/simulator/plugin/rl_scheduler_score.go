package plugin

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"

	simontype "github.com/hkust-adsl/kubernetes-scheduler-simulator/pkg/type"
)

// RLSchedulerScorePlugin queries an external RL service for node scores.
type RLSchedulerScorePlugin struct {
	handle   framework.Handle
	client   *http.Client
	endpoint string
}

var _ framework.ScorePlugin = &RLSchedulerScorePlugin{}
var _ framework.PreScorePlugin = &RLSchedulerScorePlugin{}

const rlScoreStateKey = "PreScore-RLSchedulerScorePlugin"

// rlScoreState stores node scores returned by the RL service.
type rlScoreState struct {
	scores map[string]int64
}

// Clone implements the StateData interface.
func (s *rlScoreState) Clone() framework.StateData { return s }

// NewRLSchedulerScorePlugin initializes the plugin.
func NewRLSchedulerScorePlugin(configuration runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	endpoint := os.Getenv("RL_SCHEDULER_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:5000/score"
	}
	return &RLSchedulerScorePlugin{
		handle:   handle,
		client:   &http.Client{},
		endpoint: endpoint,
	}, nil
}

// Name returns the plugin name.
func (p *RLSchedulerScorePlugin) Name() string {
	return simontype.RLSchedulerScorePluginName
}

// rlRequest carries the scheduling context sent to the RL service.
type rlRequest struct {
	Pod   *corev1.Pod   `json:"pod"`
	Nodes []corev1.Node `json:"nodes"`
}

// rlResponse captures scores from the RL service.
type rlResponse struct {
	Scores map[string]int64 `json:"scores"`
}

// PreScore calls the RL service once per scheduling cycle to obtain node scores.
func (p *RLSchedulerScorePlugin) PreScore(ctx context.Context, state *framework.CycleState, pod *corev1.Pod, nodes []*corev1.Node) *framework.Status {
	req := rlRequest{Pod: pod.DeepCopy()}
	for _, n := range nodes {
		req.Nodes = append(req.Nodes, *n)
	}

	payload, err := json.Marshal(req)
	if err != nil {
		return framework.AsStatus(err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, bytes.NewReader(payload))
	if err != nil {
		return framework.AsStatus(err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return framework.AsStatus(err)
	}
	defer resp.Body.Close()

	var r rlResponse
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return framework.AsStatus(err)
	}

	state.Write(rlScoreStateKey, &rlScoreState{scores: r.Scores})
	return nil
}

// Score returns the score for a given node from the RL service response.
func (p *RLSchedulerScorePlugin) Score(ctx context.Context, state *framework.CycleState, pod *corev1.Pod, nodeName string) (int64, *framework.Status) {
	c, err := state.Read(rlScoreStateKey)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("reading %q from cycleState: %w", rlScoreStateKey, err))
	}
	s, ok := c.(*rlScoreState)
	if !ok {
		return 0, framework.AsStatus(fmt.Errorf("cannot convert saved state to rlScoreState"))
	}
	if score, ok := s.scores[nodeName]; ok {
		return score, framework.NewStatus(framework.Success)
	}
	return framework.MinNodeScore, framework.NewStatus(framework.Success)
}

// ScoreExtensions is not used.
func (p *RLSchedulerScorePlugin) ScoreExtensions() framework.ScoreExtensions { return nil }
