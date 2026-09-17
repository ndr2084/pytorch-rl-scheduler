package proto

// Hand-written gRPC service plumbing for RLScorer.
//
// protoc-gen-go-grpc (the usual codegen for this file) targets a newer grpc-go
// API (grpc.ClientConnInterface / grpc.ServiceRegistrar) than the version this
// repo vendors (google.golang.org/grpc v1.26.0, pinned via a replace directive
// for k8s.io/kubernetes compatibility). This file implements the same client/
// server contract by hand against the stable, version-agnostic low-level API
// (*grpc.ClientConn.Invoke, *grpc.Server.RegisterService) so it builds against
// the pinned grpc version. If the vendored grpc version is ever upgraded past
// v1.29, this file can be deleted and regenerated with protoc-gen-go-grpc.

import (
	"context"

	"google.golang.org/grpc"
)

const (
	rlScorerScorePlacementMethod = "/rlscore.RLScorer/ScorePlacement"
	rlScorerScoreBatchMethod     = "/rlscore.RLScorer/ScoreBatch"
)

// RLScorerClient is the client API for the RLScorer service.
type RLScorerClient interface {
	ScorePlacement(ctx context.Context, in *ScoreRequest, opts ...grpc.CallOption) (*ScoreResponse, error)
	ScoreBatch(ctx context.Context, in *BatchScoreRequest, opts ...grpc.CallOption) (*BatchScoreResponse, error)
}

type rlScorerClient struct {
	cc *grpc.ClientConn
}

// NewRLScorerClient wraps an existing connection to the RL inference sidecar.
func NewRLScorerClient(cc *grpc.ClientConn) RLScorerClient {
	return &rlScorerClient{cc: cc}
}

func (c *rlScorerClient) ScorePlacement(ctx context.Context, in *ScoreRequest, opts ...grpc.CallOption) (*ScoreResponse, error) {
	out := new(ScoreResponse)
	if err := c.cc.Invoke(ctx, rlScorerScorePlacementMethod, in, out, opts...); err != nil {
		return nil, err
	}
	return out, nil
}

func (c *rlScorerClient) ScoreBatch(ctx context.Context, in *BatchScoreRequest, opts ...grpc.CallOption) (*BatchScoreResponse, error) {
	out := new(BatchScoreResponse)
	if err := c.cc.Invoke(ctx, rlScorerScoreBatchMethod, in, out, opts...); err != nil {
		return nil, err
	}
	return out, nil
}

// RLScorerServer is the server API for the RLScorer service. Implement this
// in a Go reference/test server; a real policy sidecar is typically Python
// and only needs to satisfy the wire contract in rlscore.proto.
type RLScorerServer interface {
	ScorePlacement(context.Context, *ScoreRequest) (*ScoreResponse, error)
	ScoreBatch(context.Context, *BatchScoreRequest) (*BatchScoreResponse, error)
}

// RegisterRLScorerServer registers srv on s so it can be reached via the
// RLScorer service methods used by RLScorerClient.
func RegisterRLScorerServer(s *grpc.Server, srv RLScorerServer) {
	s.RegisterService(&rlScorerServiceDesc, srv)
}

func rlScorerScorePlacementHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ScoreRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RLScorerServer).ScorePlacement(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: rlScorerScorePlacementMethod}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RLScorerServer).ScorePlacement(ctx, req.(*ScoreRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func rlScorerScoreBatchHandler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(BatchScoreRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RLScorerServer).ScoreBatch(ctx, in)
	}
	info := &grpc.UnaryServerInfo{Server: srv, FullMethod: rlScorerScoreBatchMethod}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RLScorerServer).ScoreBatch(ctx, req.(*BatchScoreRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var rlScorerServiceDesc = grpc.ServiceDesc{
	ServiceName: "rlscore.RLScorer",
	HandlerType: (*RLScorerServer)(nil),
	Methods: []grpc.MethodDesc{
		{MethodName: "ScorePlacement", Handler: rlScorerScorePlacementHandler},
		{MethodName: "ScoreBatch", Handler: rlScorerScoreBatchHandler},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "pkg/rlscore/proto/rlscore.proto",
}
