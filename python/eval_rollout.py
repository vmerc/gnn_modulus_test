from dataclasses import dataclass
from typing import Optional

import torch

from modulus.launch.utils import load_checkpoint

from python.CustomMeshGraphNet import MeshGraphNet


@dataclass(frozen=True)
class RolloutContext:
    device: torch.device
    dyn_start: int
    dyn_len: int
    state_mean: torch.Tensor
    state_std: torch.Tensor
    delta_mean: torch.Tensor
    delta_std: torch.Tensor


@dataclass
class RolloutState:
    graph: object
    static_part: torch.Tensor
    xn_t_full: torch.Tensor
    q_mask: torch.Tensor
    h_mask: torch.Tensor


@dataclass
class RolloutStepResult:
    prediction_normalized: torch.Tensor
    predicted_state: torch.Tensor
    target_state: torch.Tensor
    next_state: RolloutState


def build_model(
    num_input_features: int,
    num_edge_features: int,
    num_output_features: int,
    mp_layers: int,
    do_concat_trick: bool,
    num_processor_checkpoint_segments: int,
    hidden_dim_processor: int = 64,
    hidden_dim_node_encoder: int = 64,
    hidden_dim_edge_encoder: int = 64,
    hidden_dim_node_decoder: int = 64,
) -> MeshGraphNet:
    return MeshGraphNet(
        num_input_features,
        num_edge_features,
        num_output_features,
        processor_size=mp_layers,
        hidden_dim_processor=hidden_dim_processor,
        hidden_dim_node_encoder=hidden_dim_node_encoder,
        hidden_dim_edge_encoder=hidden_dim_edge_encoder,
        hidden_dim_node_decoder=hidden_dim_node_decoder,
        do_concat_trick=do_concat_trick,
        num_processor_checkpoint_segments=num_processor_checkpoint_segments,
    )


def load_model_checkpoint(
    model: MeshGraphNet,
    ckpt_dir: str,
    epoch: int,
    device: torch.device,
) -> MeshGraphNet:
    load_checkpoint(ckpt_dir, models=model, device=device, epoch=epoch)
    model.to(device)
    model.eval()
    return model


def build_rollout_context(ds, device: torch.device, use_q_feature: bool) -> RolloutContext:
    stats = ds.node_stats
    return RolloutContext(
        device=device,
        dyn_start=ds.base_graph.ndata["static"].shape[1],
        dyn_len=4 if use_q_feature else 3,
        state_mean=torch.tensor(
            [stats["h"].item(), stats["u"].item(), stats["v"].item()],
            device=device,
        ),
        state_std=torch.tensor(
            [stats["h_std"].item(), stats["u_std"].item(), stats["v_std"].item()],
            device=device,
        ),
        delta_mean=torch.tensor(
            [stats["delta_h"].item(), stats["delta_u"].item(), stats["delta_v"].item()],
            device=device,
        ),
        delta_std=torch.tensor(
            [
                stats["delta_h_std"].item(),
                stats["delta_u_std"].item(),
                stats["delta_v_std"].item(),
            ],
            device=device,
        ),
    )


def denormalize_state(xn: torch.Tensor, context: RolloutContext) -> torch.Tensor:
    return xn * context.state_std + context.state_mean


def denormalize_delta(yn: torch.Tensor, context: RolloutContext) -> torch.Tensor:
    return yn * context.delta_std + context.delta_mean


def renormalize_state(x: torch.Tensor, context: RolloutContext) -> torch.Tensor:
    return (x - context.state_mean) / (context.state_std + 1e-12)


def _build_boundary_masks(static_part: torch.Tensor, device: torch.device):
    onehot = static_part[:, :4]
    q_mask = (onehot == torch.tensor([0, 0, 1, 0], device=device)).all(dim=1)
    h_mask = (onehot == torch.tensor([0, 1, 0, 0], device=device)).all(dim=1)
    return q_mask, h_mask


def create_rollout_state(graph, context: RolloutContext) -> RolloutState:
    graph = graph.to(context.device)
    static_part = graph.ndata["x"][:, :context.dyn_start]
    xn_t_full = graph.ndata["x"][:, context.dyn_start:context.dyn_start + context.dyn_len]
    q_mask, h_mask = _build_boundary_masks(static_part, context.device)
    return RolloutState(
        graph=graph,
        static_part=static_part,
        xn_t_full=xn_t_full,
        q_mask=q_mask,
        h_mask=h_mask,
    )


def apply_boundary_conditions(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    q_mask: torch.Tensor,
    h_mask: torch.Tensor,
) -> torch.Tensor:
    x_pred = x_pred.clone()
    x_pred[q_mask] = x_gt[q_mask]
    x_pred[h_mask, 0:1] = x_gt[h_mask, 0:1]
    return x_pred


def _build_next_dynamic_features(
    state: RolloutState,
    context: RolloutContext,
    x_pred: torch.Tensor,
    q_t1_n: Optional[torch.Tensor],
) -> torch.Tensor:
    xn_t1 = renormalize_state(x_pred, context)
    if context.dyn_len == 4:
        if q_t1_n is None:
            q_t1_n = state.xn_t_full[:, 3:4]
        return torch.cat([xn_t1, q_t1_n], dim=1)
    return xn_t1


def rollout_step(
    model,
    state: RolloutState,
    context: RolloutContext,
    x_gt: torch.Tensor,
    q_t1_n: Optional[torch.Tensor] = None,
) -> RolloutStepResult:
    with torch.no_grad():
        y_pred_n = model(state.graph.ndata["x"], state.graph.edata["x"], state.graph)

    x_t = denormalize_state(state.xn_t_full[:, :3], context)
    y_pred = denormalize_delta(y_pred_n, context)
    x_pred = apply_boundary_conditions(x_t + y_pred, x_gt, state.q_mask, state.h_mask)

    next_xn_t_full = _build_next_dynamic_features(state, context, x_pred, q_t1_n)
    next_graph = state.graph.clone()
    next_graph.ndata["x"] = torch.cat([state.static_part, next_xn_t_full], dim=1)
    next_state = RolloutState(
        graph=next_graph,
        static_part=state.static_part,
        xn_t_full=next_xn_t_full,
        q_mask=state.q_mask,
        h_mask=state.h_mask,
    )

    return RolloutStepResult(
        prediction_normalized=y_pred_n,
        predicted_state=x_pred,
        target_state=x_gt,
        next_state=next_state,
    )
