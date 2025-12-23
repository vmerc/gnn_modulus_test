import os
import sys
import time
import random
import torch
import torch.nn.functional as F
import dgl
import hydra
import numpy as np

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

# repo root (ajuste si besoin)
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
if project_path not in sys.path:
    sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDataset
from python.CustomMeshGraphNet import MeshGraphNet

from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- collate: on renvoie la séquence telle quelle (liste de graphs) ---
def collate_sequence(batch):
    # batch: liste d'items; chaque item = [g_t0, g_t1, ..., g_tK]
    # On veut regrouper par temps -> on batch chaque pas t
    seq_len = len(batch[0])
    batched_graphs = []
    for t in range(seq_len):
        gt_list = [seq[t] for seq in batch]
        batched_graphs.append(dgl.batch(gt_list))
    return batched_graphs  # liste de DGLGraphs (len = seq_len)


class MGNTrainer:
    def __init__(self, cfg: DictConfig, r0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp  = bool(cfg.amp)

        # === Dataset ===
        # IMPORTANT : sequence_length >= 2 pour le pushforward (K = seq_len-1)
        self.sequence_length = int(cfg.sequence_length)
        assert self.sequence_length >= 2, "sequence_length doit être >= 2 pour le pushforward."

        dataset = TelemacDataset(
            name="telemac_train",
            data_dir=to_absolute_path(cfg.data_dir),
            dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
            split="train",
            ckpt_path=to_absolute_path(cfg.ckpt_path),
            normalize=True,
            sequence_length=self.sequence_length,
            overlap=self.sequence_length,  # 1 séquence par pas de temps (stride=1)
        )

        # === DataLoader ===
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_sequence,
        )

        # === Model ===
        self.model = MeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            processor_size=cfg.mp_layers,
            hidden_dim_processor=64,
            hidden_dim_node_encoder=64,
            hidden_dim_edge_encoder=64,
            hidden_dim_node_decoder=64,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
        ).to(self.dist.device)

        if bool(cfg.jit):
            if not getattr(self.model, "meta", None) or not getattr(self.model.meta, "jit", False):
                raise ValueError("MeshGraphNet n'est pas JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)

        # DDP
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()

        # === Optim & Loss & Scheduler ===
        self.criterion = torch.nn.MSELoss()
        self.optimizer = None
        try:
            if bool(cfg.use_apex):
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            r0.warning("Apex non installé, utilisation d'Adam standard.")
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        r0.info(f"Optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: cfg.lr_decay_rate**e
        )
        self.scaler = GradScaler(enabled=self.amp)

        # === Checkpoint load ===
        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        # point de départ du pushforward (pour le schedule TF)
        self.pf_start_epoch   = self.epoch_init
        self.pf_warmup_epochs = int(getattr(cfg, "pf_warmup_epochs", 0))
        # proba de teacher-forcing décroissante à partir de pf_start_epoch
        self.p_tf = (
            (lambda ep:
                0.0 if self.pf_warmup_epochs <= 0
                else max(0.0, 1.0 - max(0, ep - self.pf_start_epoch)/self.pf_warmup_epochs))
        )

        # === Infos normalisation (pour dé/renorm) ===
        ns = dataset.node_stats
        # on stocke des tenseurs sur device pour éviter des conversions répétées
        self.mx = torch.tensor([ns['h'        ].item(), ns['u'        ].item(), ns['v'        ].item()],
                               dtype=torch.float32, device=self.dist.device)
        self.sx = torch.tensor([ns['h_std'    ].item(), ns['u_std'    ].item(), ns['v_std'    ].item()],
                               dtype=torch.float32, device=self.dist.device)
        self.my = torch.tensor([ns['delta_h'  ].item(), ns['delta_u'  ].item(), ns['delta_v'  ].item()],
                               dtype=torch.float32, device=self.dist.device)
        self.sy = torch.tensor([ns['delta_h_std'].item(), ns['delta_u_std'].item(), ns['delta_v_std'].item()],
                               dtype=torch.float32, device=self.dist.device)

        # indices dynamiques
        self.DYN_START = dataset.base_graph.ndata['static'].shape[1]  # 6 (onehot4 + strickler + z)
        self.DYN_LEN   = 3

        # === Hyperparamètres "front d'eau" (Dice) ===
        self.lambda_front = float(getattr(cfg, "lambda_front", 0.1))  # poids de la Dice
        self.eps_front    = float(getattr(cfg, "eps_front", 0.05))    # seuil h (m) pour binaire inondation
        self.tau_front    = float(getattr(cfg, "tau_front", 0.02))    # température sigmoïde
        self.exclude_bc_in_front = bool(getattr(cfg, "exclude_bc_in_front", True))

    @staticmethod
    def _denorm(xn, mean, std):
        # xn, mean, std: tensors sur le même device
        return xn * std + mean

    @staticmethod
    def _renorm(x, mean, std):
        return (x - mean) / (std + 1e-12)

    def _apply_bc_inplace(self, x_pred, x_gt, onehot):
        # x_pred, x_gt: (N,3) sur device (non normalisés)
        q_mask = (onehot == torch.tensor([0,0,1,0], device=onehot.device)).all(dim=1)
        h_mask = (onehot == torch.tensor([0,1,0,0], device=onehot.device)).all(dim=1)
        # Q-prescrit : on remplace (h,u,v)
        x_pred[q_mask] = x_gt[q_mask]
        # H-prescrit : on remplace seulement h
        x_pred[h_mask, 0] = x_gt[h_mask, 0]
        return x_pred

    def _dice_front_loss(self, h_pred, h_gt, onehot):
        """
        Dice loss entre proba d'inondation prédite et masque binaire GT.
        h_pred, h_gt: (N,) non normalisés (mètres), CL déjà appliquées.
        onehot: (N,4) pour exclure nœuds à CL si demandé.
        """
        device = h_pred.device
        if self.exclude_bc_in_front:
            q_mask = (onehot == torch.tensor([0,0,1,0], device=device)).all(dim=1)
            h_mask = (onehot == torch.tensor([0,1,0,0], device=device)).all(dim=1)
            valid = ~(q_mask | h_mask)
        else:
            valid = torch.ones_like(h_pred, dtype=torch.bool, device=device)

        # proba d'inondation lissée
        logits = (h_pred - self.eps_front) / max(self.tau_front, 1e-12)
        p = torch.sigmoid(logits)

        # binaire GT
        m = (h_gt >= self.eps_front).float()

        # applique valid
        p = p[valid]
        m = m[valid]

        smooth = 1e-6
        inter  = (p * m).sum()
        denom  = p.sum() + m.sum()
        dice   = 1.0 - (2.0 * inter + smooth) / (denom + smooth)

        if torch.isnan(dice):
            return torch.zeros((), device=device, dtype=h_pred.dtype)
        return dice

    def train_pushforward(self, graphs, epoch):
        """
        graphs: liste [g_t0, g_t1, ..., g_tK] (batchés)
        Perte = somme des pertes one-step sur t=0..K-1 (targets normalisés)
        + lambda_front * Dice(front_h) sur x_{t+1} prédits (non normalisé).
        Réinjection = prédiction dénormalisée (avec CL), renormalisée pour t+1
        Teacher forcing stochastique (proba p_tf(epoch))
        Réinjection .detach() pour éviter BPTT long.
        """
        K = len(graphs) - 1
        assert K >= 1, "La séquence doit contenir au moins 2 pas."

        # état courant
        g = graphs[0].to(self.dist.device)
        onehot = g.ndata['x'][:, :4]  # constant sur la séquence

        total_loss = 0.0
        self.optimizer.zero_grad()

        for t in range(K):
            # --- forward one-step (targets normalisés = Δ normalisés) ---
            with autocast(enabled=self.amp):
                y_pred_n = self.model(g.ndata["x"], g.edata["x"], g)            # (N,3), normalisé
                y_gt_n   = graphs[t].ndata["y"].to(self.dist.device)            # (N,3), normalisé
                loss_t   = self.criterion(y_pred_n, y_gt_n)                     # MSE(Δ)

            # --- réinjection pushforward ---
            # x_t (non norm)
            xn_t = g.ndata['x'][:, self.DYN_START:self.DYN_START+self.DYN_LEN]
            x_t  = self._denorm(xn_t, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            # y_pred (non norm) -> x_{t+1}^{pred}
            y_pred = self._denorm(y_pred_n, self.my.to(xn_t.device), self.sy.to(xn_t.device))
            x_t1_pred = x_t + y_pred

            # GT @ t+1 (pour CL & TF)
            xn_t1_gt = graphs[t+1].ndata['x'][:, self.DYN_START:self.DYN_START+self.DYN_LEN].to(xn_t.device)
            x_t1_gt  = self._denorm(xn_t1_gt, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            # conditions aux limites
            x_t1_pred = self._apply_bc_inplace(x_t1_pred, x_t1_gt, onehot)

            # --- Dice front loss sur h ---
            front_loss_t = self._dice_front_loss(
                h_pred = x_t1_pred[:, 0].contiguous(),
                h_gt   = x_t1_gt[:,   0].contiguous(),
                onehot = onehot
            )

            # combine (hors autocast pour maîtriser la somme en FP32)
            total_loss = total_loss + loss_t + self.lambda_front * front_loss_t

            # teacher forcing
            use_tf = (torch.rand(1, device=xn_t.device).item() < self.p_tf(epoch))
            x_t1_next = x_t1_gt if use_tf else x_t1_pred

            # renormalise & réinjecte (DETACH pour casser le graphe)
            xn_t1_next = self._renorm(x_t1_next, self.mx.to(xn_t.device), self.sx.to(xn_t.device))
            x_next_full = torch.cat([g.ndata['x'][:, :self.DYN_START], xn_t1_next], dim=1).detach()

            # reconstruire un graph identique pour t+1
            g = dgl.graph(g.edges(), num_nodes=g.num_nodes(), device=g.device)
            g.ndata['x'] = x_next_full
            g.edata['x'] = graphs[0].edata['x'].to(g.device)

        # --- backward / step ---
        if self.amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_loss.detach()

@hydra.main(version_base="1.3", config_path="conf", config_name=None)
def main(cfg: DictConfig):
    # Init distrib
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("train")
    r0 = RankZeroLoggingWrapper(logger, dist)
    r0.file_logging()

    base_seed = int(getattr(cfg, "seed", 0))
    seed_everything(base_seed + int(dist.rank))
    r0.info(f"Seed: {base_seed} (rank-adjusted: {base_seed + int(dist.rank)})")

    trainer = MGNTrainer(cfg, r0)

    start = time.time()
    r0.info(f"len(dataloader) = {len(trainer.dataloader)}")
    r0.info("Training (pushforward + Dice front) started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        for graphs in trainer.dataloader:  # graphs = liste [g_t0, ..., g_tK] batched
            loss = trainer.train_pushforward(graphs, epoch)
            epoch_loss += loss.item()
        epoch_loss /= len(trainer.dataloader)

        # scheduler -> par epoch
        trainer.scheduler.step()

        r0.info(
            f"epoch: {epoch}, p_tf={trainer.p_tf(epoch):.3f}, "
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e}, "
            f"loss: {epoch_loss:10.3e}, time/epoch: {(time.time()-start):.2f}s"
        )
        start = time.time()

        # checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if (dist.rank == 0) and (epoch % 10 == 0):
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")

    r0.info("Training completed!")


if __name__ == "__main__":
    # Ex.: python train_pushforward.py <config_name>
    if len(sys.argv) < 2:
        print("Usage: python train_pushforward.py <config_name>")
        sys.exit(1)

    config_name = sys.argv.pop(1)
    if config_name.endswith((".yaml", ".yml")):
        config_name = os.path.splitext(config_name)[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conf_dir = os.path.join(script_dir, "conf")
    with hydra.initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = hydra.compose(config_name=config_name)
        main(cfg)
