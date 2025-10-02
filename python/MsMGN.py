import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

try:
    import dgl  # noqa: F401 for docs
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

# Importations nécessaires de Modulus et des blocs de MeshGraphNet
from modulus.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.gnn_layers.mesh_node_block import MeshNodeBlock
from modulus.models.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn
from modulus.models.layers import get_activation
from modulus.models.meta import ModelMetaData
from modulus.models.module import Module

# On suppose que la classe MeshGraphNetProcessor est définie comme dans le code d'origine
class MeshGraphNetProcessor(nn.Module):
    """MeshGraphNet processor block"""

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: nn.Module = nn.PReLU(),
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments

        edge_block_invars = (
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_edge,
            activation_fn,
            norm_type,
            do_concat_trick,
            False,
        )
        node_block_invars = (
            aggregation,
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_node,
            activation_fn,
            norm_type,
            False,
        )

        edge_blocks = [
            MeshEdgeBlock(*edge_block_invars) for _ in range(self.processor_size)
        ]
        node_blocks = [
            MeshNodeBlock(*node_block_invars) for _ in range(self.processor_size)
        ]
        # Intercaler les blocs d'arêtes et de nœuds
        layers = []
        for e_block, n_block in zip(edge_blocks, node_blocks):
            layers.append(e_block)
            layers.append(n_block)
        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        self.set_checkpoint_segments(self.num_processor_checkpoint_segments)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def run_function(
        self, segment_start: int, segment_end: int
    ) -> Callable[
        [Tensor, Tensor, Union[DGLGraph, List[DGLGraph]]], Tuple[Tensor, Tensor]
    ]:
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            edge_features: Tensor,
            graph: Union[DGLGraph, List[DGLGraph]],
        ) -> Tuple[Tensor, Tensor]:
            for module in segment:
                edge_features, node_features = module(
                    edge_features, node_features, graph
                )
            return edge_features, node_features

        return custom_forward

    @torch.jit.unused
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
    ) -> Tensor:
        for segment_start, segment_end in self.checkpoint_segments:
            edge_features, node_features = self.checkpoint_fn(
                self.run_function(segment_start, segment_end),
                node_features,
                edge_features,
                graph,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return node_features


@dataclass
class MetaData(ModelMetaData):
    name: str = "MultiScaleMeshGraphNet"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    onnx: bool = False
    func_torch: bool = True
    auto_grad: bool = True


class MultiScaleMeshGraphNet(Module):
    """
    Réseau MeshGraphNet multi-échelle prenant en entrée 4 graphes distincts :
      - graph_fine : graphe du maillage fin,
      - graph_coarse : graphe du maillage grossier (statique),
      - graph_down : graphe de downscaling (fine → grossier),
      - graph_up : graphe d'upscaling (grossier → fin).

    Les tenseurs d'entrées sont :
      - node_features_fine : caractéristiques des nœuds fins,
      - node_features_coarse : caractéristiques des nœuds grossiers,
      - edge_features_fine : caractéristiques des arêtes fines,
      - edge_features_coarse : caractéristiques des arêtes grossières.
    """

    def __init__(
        self,
        input_dim_nodes_fine: int,
        input_dim_nodes_coarse: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size_fine: int = 3,
        processor_size_coarse: int = 3,
        processor_size_trans: int = 1,

        mlp_activation_fn: Union[str, List[str]] = "selu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 30,
        hidden_dim_node_encoder: int = 30,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 30,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 30,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = True,
        num_processor_checkpoint_segments: int = 0,
    ):
        super().__init__(meta=MetaData())
        activation_fn = get_activation(mlp_activation_fn)

        # Encodeurs pour le graphe fin
        self.edge_encoder_fine = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        
        self.edge_encoder_down = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.edge_encoder_up = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.edge_encoder_coarse = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        
        self.node_encoder_fine = MeshGraphMLP(
            input_dim_nodes_fine,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )

        # Encodeur pour le graphe grossier (nœuds)
        self.node_encoder_coarse = MeshGraphMLP(
            input_dim_nodes_coarse,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=False,
        )

        # Décodage final sur les nœuds fins
        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=False,
        )

        # Processors multi-échelle
        self.processor_fin = MeshGraphNetProcessor(
            processor_size=processor_size_fine,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )
        self.processor_down = MeshGraphNetProcessor(
            processor_size=processor_size_trans,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )
        self.processor_coarse = MeshGraphNetProcessor(
            processor_size=processor_size_coarse,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )
        self.processor_up = MeshGraphNetProcessor(
            processor_size=processor_size_trans,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        )

        
    def encode_inputs(
        self,
        node_features_fine: Tensor,
        node_features_coarse: Tensor,
        edge_features_fine: Tensor,
        edge_features_coarse: Tensor,
        graph_down: DGLGraph,
        graph_up: DGLGraph,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            
        """Encode les caractéristiques d'entrée pour les graphes fin et grossier."""
            
        encoded_node_fine = self.node_encoder_fine(node_features_fine)
        encoded_edge_fine = self.edge_encoder_fine(edge_features_fine)
        encoded_node_coarse = self.node_encoder_coarse(node_features_coarse)
        encoded_edge_coarse = self.edge_encoder_coarse(edge_features_coarse)
        down_edge_features = self.edge_encoder_down(graph_down.edata["x"])
        up_edge_features = self.edge_encoder_up(graph_up.edata["x"])
            
        return (
                encoded_node_fine,
                encoded_edge_fine,
                encoded_node_coarse,
                encoded_edge_coarse,
                down_edge_features,
                up_edge_features,
            )

    def process_fine_graph(
            self, encoded_node_fine: Tensor, encoded_edge_fine: Tensor, graph_fine: DGLGraph
        ) -> Tensor:
        """Traitement initial sur le graphe fin."""
        return self.processor_fin(encoded_node_fine, encoded_edge_fine, graph_fine)

    def downscale(
        self,
        fine_x: Tensor,
        encoded_node_coarse: Tensor,
        down_edge_features: Tensor,
        graph_down: DGLGraph,
        ) -> Tensor:
        """
        Concatène les représentations fines et grossières, traite le downscaling
        et extrait explicitement les noeuds grossiers à l'aide d'indices explicites.
        """
        n_fine = fine_x.size(0)  # Nombre de nœuds fins
        n_coarse = encoded_node_coarse.size(0)  # Nombre de nœuds grossiers
        down_input = torch.cat((fine_x, encoded_node_coarse), dim=0)
        x_down = self.processor_down(down_input, down_edge_features, graph_down)
        # Extraction explicite des nœuds grossiers : ils occupent les positions de n_fine à n_fine+n_coarse
        x_coarse = x_down[n_fine:n_fine + n_coarse, :]
        return x_coarse

    def upscale(
        self,
        fine_x: Tensor,
        x_coarse: Tensor,
        up_edge_features: Tensor,
        graph_up: DGLGraph,
        ) -> Tensor:
        """
        Concatène la sortie traitée du graphe grossier et les caractéristiques fines,
        puis extrait explicitement les nœuds fins à l'aide d'indices explicites.
        """
        n_coarse = x_coarse.size(0)  # Nombre de nœuds grossiers
        n_fine = fine_x.size(0)      # Nombre de nœuds fins
        up_input = torch.cat((x_coarse, fine_x), dim=0)
        x_up = self.processor_up(up_input, up_edge_features, graph_up)
        # Extraction explicite des nœuds fins : ils occupent les positions de n_coarse à n_coarse+n_fine
        fine_x_up = x_up[n_coarse:n_coarse + n_fine, :]
        return fine_x_up

    def final_processing(
        self, fine_x_up: Tensor, encoded_edge_fine: Tensor, graph_fine: DGLGraph
        ) -> Tensor:
        """Applique le traitement final sur le graphe fin et décode la sortie."""
        final_x = self.processor_fin(fine_x_up, encoded_edge_fine, graph_fine)
        output = self.node_decoder(final_x)
        return output

    def forward(self, *args, **kwargs):
        # Permet de récupérer les arguments par mot-clé ou par position
        if kwargs:
            node_features_fine = kwargs.pop('node_features_fine')
            node_features_coarse = kwargs.pop('node_features_coarse')
            edge_features_fine = kwargs.pop('edge_features_fine')
            edge_features_coarse = kwargs.pop('edge_features_coarse')
            graph_fine = kwargs.pop('graph_fine')
            graph_coarse = kwargs.pop('graph_coarse')
            graph_down = kwargs.pop('graph_down')
            graph_up = kwargs.pop('graph_up')
        else:
            (
                node_features_fine,
                node_features_coarse,
                edge_features_fine,
                edge_features_coarse,
                graph_fine,
                graph_coarse,
                graph_down,
                graph_up,
            ) = args


        # Étape 1 : Encodage des entrées
        (
            encoded_node_fine,
            encoded_edge_fine,
            encoded_node_coarse,
            encoded_edge_coarse,
            down_edge_features,
            up_edge_features,
        ) = self.encode_inputs(
            node_features_fine,
            node_features_coarse,
            edge_features_fine,
            edge_features_coarse,
            graph_down,
            graph_up,
        )

        # Étape 2 : Traitement initial sur le graphe fin
        fine_x = self.process_fine_graph(encoded_node_fine, encoded_edge_fine, graph_fine)

        # Étape 3 : Downscaling et extraction explicite des nœuds grossiers
        x_coarse = self.downscale(fine_x, encoded_node_coarse, down_edge_features, graph_down)

        # Traitement sur le graphe grossier
        x_coarse = self.processor_coarse(x_coarse, encoded_edge_coarse, graph_coarse)

        # Étape 4 : Upscaling et extraction explicite des nœuds fins
        fine_x_up = self.upscale(fine_x, x_coarse, up_edge_features, graph_up)

        # Étape 5 : Traitement final et décodage
        output = self.final_processing(fine_x_up, encoded_edge_fine, graph_fine)
        return output
