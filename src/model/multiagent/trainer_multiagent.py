import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from src.metrics import AvgMinADE, AvgMinFDE, ActorMR
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2_multiagent import SubmissionAv2MultiAgent

from src.model.multiagent.model_multiagent import ModelMultiAgent
from src.model.layers.multihead import myMultiheadAttention


class Trainer(pl.LightningModule):

    def __init__(
            self,
            dim=128,
            historical_steps=50,
            future_steps=60,
            embedding_type="fourier",
            encoder_depth=4,
            spa_depth=3,
            decoder_depth=3,
            scene_score_depth=2,
            num_heads=8,
            attn_bias=True,
            ffn_bias=True,
            dropout=0.1,
            num_modes=6,
            act_layer=nn.ReLU,
            norm_layer=nn.LayerNorm,
            use_cls_token=True,
            lr: float = 1e-3,
            warmup_epochs: int = 10,
            epochs: int = 60,
            weight_decay: float = 1e-4,
            submission_handler=SubmissionAv2MultiAgent(),
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["submission_handler"])
        self.num_modes = num_modes
        act_layer = eval(act_layer)
        norm_layer = eval(norm_layer)

        self.net = ModelMultiAgent(
            embed_dim=dim,
            embedding_type=embedding_type,
            encoder_depth=encoder_depth,
            spa_depth=spa_depth,
            decoder_depth=decoder_depth,
            scene_score_depth=scene_score_depth,
            num_heads=num_heads,
            attn_bias=attn_bias,
            ffn_bias=ffn_bias,
            dropout=dropout,
            future_steps=future_steps,
            num_modes=num_modes,
            use_cls_token=use_cls_token,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        # self.net = torch.compile(self.net)

        metrics = MetricCollection({
            "AvgMinADE1": AvgMinADE(k=1),
            "AvgMinADE": AvgMinADE(k=6),
            "AvgMinFDE1": AvgMinFDE(k=1),
            "AvgMinFDE": AvgMinFDE(k=6),
            "ActorMR": ActorMR(),
        })
        self.val_metrics = metrics.clone(prefix="val_")
        self.submission_handler = submission_handler

    def forward(self, data):
        return self.net(data)

    def cal_loss(self, outputs, data):
        y_hat = outputs["y_hat"]
        pi = outputs["pi"]
        y_propose = outputs["y_propose"]

        x_scored, y, y_padding_mask = (
            data["x_scored"],
            data["y"],
            data["x_padding_mask"][..., 50:],
        )

        # TODO: only consider scored agents
        valid_mask = ~y_padding_mask
        valid_mask[~x_scored] = False  # [b,n,t]
        valid_mask = valid_mask.unsqueeze(2).float()  # [b,n,1,t]

        scene_avg_ade = (torch.norm(y_hat[..., :2] - y.unsqueeze(2), dim=-1) *
                         valid_mask).sum(dim=(-1, -3)) / valid_mask.sum(
                             dim=(-1, -3))
        best_mode = torch.argmin(scene_avg_ade, dim=-1)
        y_hat_best = y_hat[
            # torch.arange(y_hat.shape[0]).unsqueeze(1),
            torch.arange(y_hat.shape[0]),
            # torch.arange(y_hat.shape[1]).unsqueeze(0),
            :,
            best_mode,
            :,
            :,
        ]
        y_propose_best = y_propose[
            # torch.arange(y_propose.shape[0]).unsqueeze(1),
            torch.arange(y_propose.shape[0]),
            # torch.arange(y_propose.shape[1]).unsqueeze(0),
            :,
            best_mode,
            :,
            :,
        ]
        reg_mask = ~y_padding_mask  # [b,n,t]
        reg_mask[~x_scored] = False
        # y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        reg_loss = F.smooth_l1_loss(y_hat_best[reg_mask], y[reg_mask])
        propose_reg_loss = F.smooth_l1_loss(y_propose_best[reg_mask],
                                            y[reg_mask])
        # cls_loss = F.cross_entropy(
        #     pi.view(-1, pi.size(-1))[reg_mask.all(-1).view(-1)],
        #     best_mode.view(-1)[reg_mask.all(-1).view(-1)].detach())
        cls_loss = F.cross_entropy(pi.squeeze(-1), best_mode.detach())

        loss = reg_loss + cls_loss + propose_reg_loss
        out = {
            "loss": loss,
            "reg_loss": reg_loss.item(),
            "cls_loss": cls_loss.item(),
            "propose_reg_loss": propose_reg_loss.item(),
        }

        return out

    def training_step(self, data, batch_idx):
        outputs = self(data)
        res = self.cal_loss(outputs, data)

        for k, v in res.items():
            if k.endswith("loss"):
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=outputs["y_hat"].shape[0],
                )

        return res["loss"]

    def validation_step(self, data, batch_idx):
        outputs = self(data)
        res = self.cal_loss(outputs, data)
        metrics = self.val_metrics(outputs, data["y"], data["x_scored"])

        for k, v in res.items():
            if k.endswith("loss"):
                self.log(
                    f"val/{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=outputs["y_hat"].shape[0],
                )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, data, batch_idx) -> None:
        outputs = self(data)
        y_hat, pi = outputs["y_hat"], outputs["pi"]
        y_hat = y_hat.permute(0, 2, 1, 3, 4)
        pi = pi.squeeze(-1)

        bs, k, n, t, _ = y_hat.shape
        centers = data["x_centers"].view(bs, 1, n, 1, 2)
        y_hat += centers

        self.submission_handler.format_data(data,
                                            y_hat,
                                            pi,
                                            normalized_probability=False)

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell,
            myMultiheadAttention,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = ("%s.%s" % (module_name, param_name)
                                   if module_name else param_name)
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param
            for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params":
                [param_dict[param_name] for param_name in sorted(list(decay))],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name]
                    for param_name in sorted(list(no_decay))
                ],
                "weight_decay":
                0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        # scheduler = WarmupCosLR(
        #     optimizer=optimizer,
        #     lr=self.lr,
        #     min_lr=1e-6,
        #     warmup_epochs=self.warmup_epochs,
        #     epochs=self.epochs,
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.epochs, eta_min=0)
        return [optimizer], [scheduler]
