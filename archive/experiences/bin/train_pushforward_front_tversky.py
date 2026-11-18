import os
import sys
import time
import random
import torch
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

from python.create_dgl_dataset import TelemacDataset, TelemacDatasetWithQ
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
        self.amp = bool(getattr(cfg, "amp", False)) and self.dist.device.type == "cuda"

        # === Dataset ===
        # IMPORTANT : sequence_length >= 2 pour le pushforward (K = seq_len-1)
        self.sequence_length = int(cfg.sequence_length)
        assert self.sequence_length >= 2, "sequence_length doit être >= 2 pour le pushforward."

        self.use_q_feature = bool(getattr(cfg, "use_q_feature", False))
        if self.use_q_feature:
            hydro_files = getattr(cfg, "hydro_dir", None)
            if hydro_files is None:
                raise ValueError("use_q_feature=True requires hydro_dir in config.")
            dataset = TelemacDatasetWithQ(
                name="telemac_train_q",
                data_dir=to_absolute_path(cfg.data_dir),
                dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
                hydro_data_files=[to_absolute_path(p) for p in hydro_files],
                split="train",
                ckpt_path=to_absolute_path(cfg.ckpt_path),
                normalize=True,
                sequence_length=self.sequence_length,
                overlap=self.sequence_length,
                dt_seconds=float(getattr(cfg, "dt_seconds", 1800.0)),
            )
        else:
            dataset = TelemacDataset(
                name="telemac_train",
                data_dir=to_absolute_path(cfg.data_dir),
                dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
                split="train",
                ckpt_path=to_absolute_path(cfg.ckpt_path),
                normalize=True,
                sequence_length=self.sequence_length,
                overlap=self.sequence_length,
            )

        expected_input_features = dataset.base_graph.ndata['static'].shape[1] + (4 if self.use_q_feature else 3)
        if int(cfg.num_input_features) != int(expected_input_features):
            raise ValueError(
                f"num_input_features={cfg.num_input_features} incompatible with dataset ({expected_input_features})."
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
        self.DYN_LEN   = 4 if self.use_q_feature else 3

        # === Hyperparamètres "front d'eau" (Tversky simple, seuil unique) ===
        self.lambda_front = float(getattr(cfg, "lambda_front", 0.1))
        self.exclude_bc_in_front = bool(getattr(cfg, "exclude_bc_in_front", True))
        self.eps_front = float(getattr(cfg, "eps_front", 0.05))

        # paramètres Tversky
        self.front_temp = float(getattr(cfg, "front_temp", getattr(cfg, "tau_front", 0.03)))
        self.front_alpha = float(getattr(cfg, "front_alpha", 0.7))  # FN
        self.front_beta = float(getattr(cfg, "front_beta", 0.3))    # FP
        self.front_gamma = float(getattr(cfg, "front_gamma", 1.5))  # focalisation

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

    def _soft_wet_prob(self, h, thr):
        temp = max(self.front_temp, 1e-12)
        return torch.sigmoid((h - thr) / temp)

    def _focal_tversky_loss(self, p, g):
        eps = 1e-6
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        tversky = (tp + eps) / (tp + self.front_alpha * fn + self.front_beta * fp + eps)
        loss = (1.0 - tversky).pow(self.front_gamma)
        if torch.isnan(loss):
            return torch.zeros((), device=p.device, dtype=p.dtype)
        return loss

    def _wet_metrics(self, h_pred, h_gt, onehot):
        device = h_pred.device
        if self.exclude_bc_in_front:
            q_mask = (onehot == torch.tensor([0,0,1,0], device=device)).all(dim=1)
            h_mask = (onehot == torch.tensor([0,1,0,0], device=device)).all(dim=1)
            valid = ~(q_mask | h_mask)
        else:
            valid = torch.ones_like(h_pred, dtype=torch.bool, device=device)

        h_pred = h_pred[valid]
        h_gt = h_gt[valid]

        if h_pred.numel() == 0:
            z = torch.zeros((), device=device, dtype=h_pred.dtype)
            return z, z, z

        wet_gt = (h_gt >= self.eps_front)
        wet_pred = (h_pred >= self.eps_front)
        wet_ratio_gt = wet_gt.float().mean()
        wet_ratio_pred = wet_pred.float().mean()
        inter = (wet_gt & wet_pred).sum().float()
        union = (wet_gt | wet_pred).sum().float()
        iou = inter / (union + 1e-6)
        return wet_ratio_gt, wet_ratio_pred, iou

    def _front_loss(self, h_pred, h_gt, onehot):
        """
        Focal-Tversky (seuil unique) entre proba d'inondation prédite et masque GT.
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

        h_pred = h_pred[valid]
        h_gt = h_gt[valid]

        if h_pred.numel() == 0:
            return torch.zeros((), device=device, dtype=h_pred.dtype)

        thr = float(self.eps_front)
        p = self._soft_wet_prob(h_pred, thr)
        g = (h_gt >= thr).float()
        return self._focal_tversky_loss(p, g)

    def train_pushforward(self, graphs, epoch):
        """
        graphs: liste [g_t0, g_t1, ..., g_tK] (batchés)
        Perte = somme des pertes one-step sur t=0..K-1 (targets normalisés)
        + lambda_front * Focal-Tversky (seuil unique) sur x_{t+1} prédits (non normalisé).
        Réinjection = prédiction dénormalisée (avec CL), renormalisée pour t+1
        Teacher forcing stochastique (proba p_tf(epoch))
        Réinjection .detach() pour éviter BPTT long.
        """
        K = len(graphs) - 1
        assert K >= 1, "La séquence doit contenir au moins 2 pas."

        # état courant
        g = graphs[0].to(self.dist.device)
        onehot = g.ndata['x'][:, :4]  # constant sur la séquence

        total_loss = torch.zeros((), device=self.dist.device)
        mse_loss_sum = torch.zeros((), device=self.dist.device)
        front_loss_sum = torch.zeros((), device=self.dist.device)
        wet_gt_sum = torch.zeros((), device=self.dist.device)
        wet_pred_sum = torch.zeros((), device=self.dist.device)
        iou_sum = torch.zeros((), device=self.dist.device)
        self.optimizer.zero_grad()

        for t in range(K):
            # --- forward one-step (targets normalisés = Δ normalisés) ---
            with autocast(enabled=self.amp):
                y_pred_n = self.model(g.ndata["x"], g.edata["x"], g)            # (N,3), normalisé
                y_gt_n   = graphs[t].ndata["y"].to(self.dist.device)            # (N,3), normalisé
                loss_t   = self.criterion(y_pred_n, y_gt_n)                     # MSE(Δ)

            # --- réinjection pushforward ---
            # x_t (non norm)
            xn_t_full = g.ndata['x'][:, self.DYN_START:self.DYN_START+self.DYN_LEN]
            xn_t = xn_t_full[:, :3]
            x_t  = self._denorm(xn_t, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            # y_pred (non norm) -> x_{t+1}^{pred}
            y_pred = self._denorm(y_pred_n, self.my.to(xn_t.device), self.sy.to(xn_t.device))
            x_t1_pred = x_t + y_pred

            # GT @ t+1 (pour CL & TF)
            xn_t1_gt_full = graphs[t+1].ndata['x'][:, self.DYN_START:self.DYN_START+self.DYN_LEN].to(xn_t.device)
            xn_t1_gt = xn_t1_gt_full[:, :3]
            x_t1_gt  = self._denorm(xn_t1_gt, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            # conditions aux limites
            x_t1_pred = self._apply_bc_inplace(x_t1_pred, x_t1_gt, onehot)

            # --- Focal-Tversky front loss sur h (seuil unique) ---
            front_loss_t = self._front_loss(
                h_pred = x_t1_pred[:, 0].contiguous(),
                h_gt   = x_t1_gt[:,   0].contiguous(),
                onehot = onehot
            )

            # combine (hors autocast pour maîtriser la somme en FP32)
            mse_loss_sum = mse_loss_sum + loss_t
            front_loss_sum = front_loss_sum + front_loss_t
            total_loss = total_loss + loss_t + self.lambda_front * front_loss_t

            with torch.no_grad():
                wet_gt, wet_pred, iou = self._wet_metrics(
                    h_pred = x_t1_pred[:, 0].contiguous(),
                    h_gt   = x_t1_gt[:,   0].contiguous(),
                    onehot = onehot
                )
            wet_gt_sum = wet_gt_sum + wet_gt
            wet_pred_sum = wet_pred_sum + wet_pred
            iou_sum = iou_sum + iou

            # teacher forcing
            use_tf = (torch.rand(1, device=xn_t.device).item() < self.p_tf(epoch))
            x_t1_next = x_t1_gt if use_tf else x_t1_pred

            # renormalise & réinjecte (DETACH pour casser le graphe)
            xn_t1_next = self._renorm(x_t1_next, self.mx.to(xn_t.device), self.sx.to(xn_t.device))
            if self.use_q_feature:
                q_t1_n = xn_t1_gt_full[:, 3:4]
                xn_t1_next_full = torch.cat([xn_t1_next, q_t1_n], dim=1)
            else:
                xn_t1_next_full = xn_t1_next
            x_next_full = torch.cat([g.ndata['x'][:, :self.DYN_START], xn_t1_next_full], dim=1).detach()

            # reconstruire un graph identique pour t+1
            g = dgl.graph(g.edges(), num_nodes=g.num_nodes(), device=g.device)
            g.ndata['x'] = x_next_full
            g.edata['x'] = graphs[0].edata['x'].to(g.device)

        denom = max(1, K)
        total_loss = total_loss / denom
        mse_loss_sum = mse_loss_sum / denom
        front_loss_sum = front_loss_sum / denom

        # --- backward / step ---
        if self.amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        wet_gt_mean = wet_gt_sum / denom
        wet_pred_mean = wet_pred_sum / denom
        iou_mean = iou_sum / denom

        return (
            total_loss.detach(),
            mse_loss_sum.detach(),
            front_loss_sum.detach(),
            wet_gt_mean.detach(),
            wet_pred_mean.detach(),
            iou_mean.detach(),
        )

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
    r0.info("Training (pushforward + Focal-Tversky front) started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_front_loss = 0.0
        epoch_wet_gt = 0.0
        epoch_wet_pred = 0.0
        epoch_iou = 0.0
        for graphs in trainer.dataloader:  # graphs = liste [g_t0, ..., g_tK] batched
            loss, mse_loss, front_loss, wet_gt, wet_pred, iou = trainer.train_pushforward(graphs, epoch)
            epoch_loss += loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_front_loss += front_loss.item()
            epoch_wet_gt += wet_gt.item()
            epoch_wet_pred += wet_pred.item()
            epoch_iou += iou.item()
        epoch_loss /= len(trainer.dataloader)
        epoch_mse_loss /= len(trainer.dataloader)
        epoch_front_loss /= len(trainer.dataloader)
        epoch_wet_gt /= len(trainer.dataloader)
        epoch_wet_pred /= len(trainer.dataloader)
        epoch_iou /= len(trainer.dataloader)

        # scheduler -> par epoch
        trainer.scheduler.step()

        if epoch % 10 == 0:
            r0.info(
                f"epoch: {epoch}, p_tf={trainer.p_tf(epoch):.3f}, "
                f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e}, "
                f"loss: {epoch_loss:10.3e}, "
                f"loss_t: {epoch_mse_loss:10.3e}, "
                f"front_loss_t: {epoch_front_loss:10.3e}, "
                f"lambda*front: {(trainer.lambda_front * epoch_front_loss):10.3e}, "
                f"wet_gt: {epoch_wet_gt:8.4f}, "
                f"wet_pred: {epoch_wet_pred:8.4f}, "
                f"iou: {epoch_iou:8.4f}, "
                f"time/epoch: {(time.time()-start):.2f}s"
            )
        else:
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
    # Ex.: python train_pushforward_front_tversky.py <config_name>
    if len(sys.argv) < 2:
        print("Usage: python train_pushforward_front_tversky.py <config_name>")
        sys.exit(1)

    config_name = sys.argv.pop(1)
    if config_name.endswith((".yaml", ".yml")):
        config_name = os.path.splitext(config_name)[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conf_dir = os.path.join(script_dir, "conf")
    with hydra.initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = hydra.compose(config_name=config_name)
        main(cfg)
