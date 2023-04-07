import copy
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from rich.console import Console
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from gans.augment.adaptive_augment import AdaptiveAugment
from gans.context_manager import gradient_accumulation
from gans.coords import CoordBridge
from gans.datasets.kitti import KITTIRaw
from gans.metrics.fpd_kpd import compute_frechet_distance, compute_squared_mmd
from gans.metrics.pointnet import pretrained_pointnet
from gans.models.builder import build_discriminator, build_generator
from gans.models.loss import GANLoss
from gans.models.ops.common import filter2d
from gans.utils import (
    InfiniteSampler,
    set_requires_grad,
    sigmoid_to_tanh,
    tanh_to_sigmoid,
)


@torch.no_grad()
def ema_inplace(ema_model, new_model, decay):
    # parameters
    ema_params = dict(ema_model.named_parameters())
    new_params = dict(new_model.named_parameters())
    for key in ema_params.keys():
        ema_params[key].data.mul_(decay).add_(new_params[key], alpha=1 - decay)
    # buffers
    ema_buffers = dict(ema_model.named_buffers())
    new_buffers = dict(new_model.named_buffers())
    for key in ema_buffers.keys():
        ema_buffers[key].data.mul_(0).add_(new_buffers[key].data, alpha=1)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(self.cfg.training.rank)
        self.console = Console(quiet=self.cfg.training.rank != 0)
        self.resolution = cfg.model.generator.synthesis_kwargs.resolution

        # setup models
        self.console.log("setting up models...")
        self.G = build_generator(self.cfg.model.generator)
        self.G_ema = copy.deepcopy(self.G).eval()
        self.D = build_discriminator(self.cfg.model.discriminator)
        self.A = AdaptiveAugment(
            p_init=self.cfg.training.augment.p_init,
            p_target=self.cfg.training.augment.p_target,
            kimg=self.cfg.training.augment.kimg,
            **self.cfg.training.augment.policy,
        )
        self.coord = CoordBridge(
            num_ring=self.resolution[0],
            num_points=self.resolution[1],
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
            angle_file=f"data/coords/{cfg.dataset.name}.npy",
        ).eval()

        self.G.to(self.device)
        self.G_ema.to(self.device)
        self.D.to(self.device)
        self.A.to(self.device)
        self.coord.to(self.device)

        ddp_kwargs = dict(device_ids=[self.cfg.training.rank])
        self.G = DDP(module=self.G, broadcast_buffers=True, **ddp_kwargs)
        self.D = DDP(module=self.D, broadcast_buffers=False, **ddp_kwargs)
        self.ddp_models = (self.G, self.D)

        self.G.requires_grad_(False)
        self.G_ema.requires_grad_(False)
        self.D.requires_grad_(False)
        self.A.requires_grad_(False)
        self.coord.requires_grad_(False)

        # auxiliary inputs
        self.auxin = {}
        if "dusty_v2" in self.cfg.model.generator.arch:
            self.auxin["angle"] = self.coord.angle.repeat_interleave(
                self.cfg.training.batch_size_per_gpu, dim=0
            )

        # training dataset
        self.console.log("setting up datasets...")

        self.train_dataset = KITTIRaw(
            root=cfg.dataset.root,
            split="train",
            shape=self.resolution,
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size_per_gpu,
            sampler=InfiniteSampler(
                self.train_dataset,
                rank=self.cfg.training.rank,
                num_replicas=self.cfg.training.num_gpus,
                seed=self.cfg.random_seed + self.cfg.training.rank,
            ),
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        self.console.log(f"train imgs: {len(self.train_dataset):,}")
        self.iter_train_loader = iter(self.train_loader)

        self.val_dataset = KITTIRaw(
            root=cfg.dataset.root,
            split="val",
            shape=self.resolution,
            min_depth=cfg.dataset.min_depth,
            max_depth=cfg.dataset.max_depth,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.training.batch_size_per_gpu,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        self.console.log(f"val imgs: {len(self.val_dataset):,}")

        # loss criterion
        self.console.log("setting up losses...")
        self.console.log("adversarial loss:", self.cfg.training.gan_objective)
        self.adversarial_loss = GANLoss(self.cfg.training.gan_objective).to(self.device)
        lazy_ratio_G = 1.0
        lazy_ratio_D = 1.0
        if "gp" in self.cfg.training.loss and self.cfg.training.loss.gp > 0.0:
            self.console.log("r1 gradient penalty:", self.cfg.training.loss.gp)
            self.cfg.training.loss.gp *= self.cfg.training.lazy.gp
            lazy_ratio_D = self.cfg.training.lazy.gp / (self.cfg.training.lazy.gp + 1.0)
        if "pl" in self.cfg.training.loss and self.cfg.training.loss.pl > 0.0:
            self.console.log("path length regularization:", self.cfg.training.loss.pl)
            self.cfg.training.loss.pl *= self.cfg.training.lazy.pl
            self.pl_ema = torch.tensor(0.0, device=self.device)
            lazy_ratio_G = self.cfg.training.lazy.pl / (self.cfg.training.lazy.pl + 1.0)

        # Optimizer
        self.console.log("setting up optimizers...")
        self.optim_G = optim.Adam(
            params=self.G.parameters(),
            lr=self.cfg.training.lr.generator.alpha * lazy_ratio_G,
            betas=(
                self.cfg.training.lr.generator.beta1**lazy_ratio_G,
                self.cfg.training.lr.generator.beta2**lazy_ratio_G,
            ),
        )
        self.optim_D = optim.Adam(
            params=self.D.parameters(),
            lr=self.cfg.training.lr.discriminator.alpha * lazy_ratio_D,
            betas=(
                self.cfg.training.lr.discriminator.beta1**lazy_ratio_D,
                self.cfg.training.lr.discriminator.beta2**lazy_ratio_D,
            ),
        )

        # automatic mixed precision
        self.scaler_D = GradScaler(enabled=self.cfg.training.amp.main)
        self.scaler_G = GradScaler(enabled=self.cfg.training.amp.main)
        self.scaler_r1 = GradScaler(enabled=self.cfg.training.amp.reg)
        self.scaler_pl = GradScaler(enabled=self.cfg.training.amp.reg)
        if self.cfg.training.amp.main:
            self.console.log("amp enabled (G & D)")
        if self.cfg.training.amp.reg:
            self.console.log("amp enabled (reg)")

        # resume from checkpoints
        self.start_iteration = 0
        if self.cfg.training.resume is not None:
            state_dict = torch.load(self.cfg.training.resume, map_location="cpu")
            self.start_iteration = state_dict["step"] // self.cfg.training.batch_size
            self.G.module.load_state_dict(state_dict["G"])
            self.D.module.load_state_dict(state_dict["D"])
            self.G_ema.load_state_dict(state_dict["G_ema"])
            self.A.load_state_dict(state_dict["A"])
            self.optim_G.load_state_dict(state_dict["optim_G"])
            self.optim_D.load_state_dict(state_dict["optim_D"])
            if "pl" in self.cfg.training.loss and self.cfg.training.loss.pl > 0.0:
                self.pl_ema = state_dict["pl_ema"].to(self.device)

        self.z_fixed = self.sample_z(self.cfg.training.batch_size_per_gpu)
        self.warmup_fade_kimg = cfg.training.warmup.fade_kimg * 1e3
        self.blur_sigma = 0
        self.dropout_ratio = 0
        self.iters_to_imgs = lambda i: int(i * self.cfg.training.batch_size)
        self.val_real_feats = None

    def sample_z(self, batch_size):
        return torch.randn(
            batch_size,
            self.cfg.model.generator.mapping_kwargs.in_ch,
            device=self.device,
        )

    def fetch_reals(self, raw_batch):
        depth = raw_batch["depth"].to(self.device)
        mask = raw_batch["mask"].to(self.device)
        x = self.coord.convert(depth, "depth", "inv_depth_norm")
        x = sigmoid_to_tanh(x)  # [-1,1]
        x = mask * x + (1 - mask) * self.cfg.dataset.raydrop_const
        return {"image": x, "raydrop_mask": mask}

    def set_warmup_params(self, iteration):
        num_imgs = self.iters_to_imgs(iteration)
        self.blur_sigma = (
            max(1 - num_imgs / self.warmup_fade_kimg, 0)
            * self.cfg.training.warmup.blur_init_sigma
            if self.warmup_fade_kimg > 0
            else 0
        )
        self.dropout_ratio = (
            max(1 - num_imgs / self.warmup_fade_kimg, 0)
            * self.cfg.training.warmup.dropout_init_ratio
            if self.warmup_fade_kimg > 0
            else 0
        )

    def warmup(self, x):
        # from StyleGAN3
        blur_size = np.floor(self.blur_sigma * 3)
        if blur_size > 0:
            blur_kernel = torch.arange(-blur_size, blur_size + 1, device=x.device)
            blur_kernel = blur_kernel.div(self.blur_sigma).square().neg().exp2()
            x = filter2d(x, blur_kernel)
        if self.dropout_ratio > 0:
            ratio = torch.full_like(x, self.dropout_ratio, device=x.device)
            mask = torch.bernoulli(1 - ratio)
            x = mask * x + (1 - mask) * self.cfg.dataset.raydrop_const
        return x

    def step(self, iteration):
        self.G.train()
        self.set_warmup_params(iteration)

        scalars = defaultdict(list)
        B = self.cfg.training.batch_size_per_gpu

        # input data
        reals = self.fetch_reals(next(self.iter_train_loader))
        reals = dict((k, v.split(B)) for k, v in reals.items())
        num_accumulation = len(reals["image"])

        #############################################################
        # train G
        #############################################################
        set_requires_grad(self.G, True)
        use_real = self.cfg.training.gan_objective in ("ragan", "rahinge", "ralsgan")

        self.optim_G.zero_grad(set_to_none=True)
        for j in gradient_accumulation(num_accumulation, True, self.ddp_models):
            with autocast(self.cfg.training.amp.main):
                # sample z
                z = self.sample_z(batch_size=B)

                # forward G
                x_fake = self.G(z, **self.auxin)["image"]

                # fake
                x_fake_wup = self.warmup(x_fake)
                x_fake_aug = self.A(x_fake_wup)
                y_fake = self.D(x_fake_aug)

                # real
                x_real = reals["image"][j]
                if use_real:
                    x_real_wup = self.warmup(x_real)
                    x_real_aug = self.A(x_real_wup).detach()
                    y_real = self.D(x_real_aug)
                else:
                    y_real = None

                loss_G = 0

                # adversarial loss
                loss_GAN = self.adversarial_loss(y_real, y_fake, "G")
                loss_G += self.cfg.training.loss.gan * loss_GAN
                scalars["loss/G/adversarial"].append(loss_GAN.detach())

                loss_G /= float(num_accumulation)

            self.scaler_G.scale(loss_G).backward()

        # update G parameters
        self.scaler_G.step(self.optim_G)
        self.scaler_G.update()

        #############################################################
        # regularize G
        #############################################################

        # path length regularization
        if (
            self.cfg.training.loss.pl > 0.0
            and iteration % self.cfg.training.lazy.pl == 0
        ):
            self.optim_G.zero_grad(set_to_none=True)
            for j in gradient_accumulation(num_accumulation, True, self.ddp_models):
                with autocast(self.cfg.training.amp.reg):
                    # forward G with smaller batch
                    B_pl = B // 2
                    z_pl = self.sample_z(B_pl).requires_grad_()
                    auxin_pl = {}
                    if "dusty_v2" in self.cfg.model.generator.arch:
                        auxin_pl["angles"] = self.coord.angle.repeat_interleave(
                            B_pl, dim=0
                        )

                    # perturb images
                    fakes_pl = self.G(z_pl, **auxin_pl)

                    noise_pl = torch.randn_like(fakes_pl["image"])
                    noise_pl /= np.sqrt(np.prod(fakes_pl["image"].shape[2:]))

                    if "dusty_v2" in self.cfg.model.generator.arch:
                        input_pl = fakes_pl["styles"].requires_grad_()
                    else:
                        input_pl = z_pl.requires_grad_()

                (grads,) = torch.autograd.grad(
                    outputs=self.scaler_pl.scale((fakes_pl["image"] * noise_pl).sum()),
                    inputs=[input_pl],
                    create_graph=True,
                    only_inputs=True,
                )

                # unscale
                grads = grads / self.scaler_pl.get_scale()

                # compute |J*y|
                with autocast(self.cfg.training.amp.reg):
                    pl_lengths = grads.pow(2).sum(dim=-1)
                    pl_lengths = torch.sqrt(pl_lengths)
                    # ema of |J*y|
                    pl_ema = self.pl_ema.lerp(pl_lengths.mean(), 0.01)
                    self.pl_ema.copy_(pl_ema.detach())
                    # calculate (|J*y|-a)^2
                    pl_penalty = (pl_lengths - pl_ema).pow(2).mean()
                    loss_G = self.cfg.training.loss.pl * pl_penalty
                    loss_G += 0.0 * fakes_pl["image"][0, 0, 0, 0]
                    loss_G /= float(num_accumulation)

                    scalars["loss/G/path_length/baseline"].append(self.pl_ema.detach())
                    scalars["loss/G/path_length"].append(pl_penalty.detach())

                self.scaler_pl.scale(loss_G).backward()

            # update G parameters
            self.scaler_pl.step(self.optim_G)
            self.scaler_pl.update()

        set_requires_grad(self.G, False)

        #############################################################
        # train D
        #############################################################

        set_requires_grad(self.D, True)

        self.optim_D.zero_grad(set_to_none=True)
        for j in gradient_accumulation(num_accumulation, True, self.ddp_models):
            with autocast(self.cfg.training.amp.main):
                # sample z
                z = self.sample_z(batch_size=B)

                # forward G
                x_real = reals["image"][j]
                x_fake = self.G(z, **self.auxin)["image"]

                # warmup
                x_real_wup = self.warmup(x_real)
                x_fake_wup = self.warmup(x_fake)

                # augment
                x_real_aug = self.A(x_real_wup).detach()
                x_fake_aug = self.A(x_fake_wup).detach()

                # forward D
                y_real = self.D(x_real_aug)
                y_fake = self.D(x_fake_aug)

                self.A.cumulate(y_real)

                # adversarial loss
                loss_GAN = self.adversarial_loss(y_real, y_fake, "D")
                loss_D = self.cfg.training.loss.gan * loss_GAN
                loss_D /= float(num_accumulation)

                scalars["loss/D/output/real"].append(y_real.mean().detach())
                scalars["loss/D/output/fake"].append(y_fake.mean().detach())
                scalars["loss/D/adversarial"].append(loss_GAN.detach())

            self.scaler_D.scale(loss_D).backward()

        # update D parameters
        self.scaler_D.step(self.optim_D)
        self.scaler_D.update()

        #############################################################
        # regularize D
        #############################################################

        # r1 gradient penalty
        if (
            self.cfg.training.loss.gp > 0.0
            and iteration % self.cfg.training.lazy.gp == 0
        ):
            self.optim_D.zero_grad(set_to_none=True)
            for j in gradient_accumulation(num_accumulation, True, self.ddp_models):
                with autocast(self.cfg.training.amp.reg):
                    input_gp = reals["image"][j].detach().requires_grad_()
                    y_real = self.D(self.A(self.warmup(input_gp)))

                # this part causes a warning about the discriminator's grads
                (grads,) = torch.autograd.grad(
                    outputs=[self.scaler_r1.scale(y_real.sum())],
                    inputs=[input_gp],
                    create_graph=True,
                )

                # unscale
                grads = grads / self.scaler_r1.get_scale()

                with autocast(self.cfg.training.amp.reg):
                    r1_penalty = (grads**2).sum(dim=[1, 2, 3]).mean()
                    loss_D = (self.cfg.training.loss.gp / 2) * r1_penalty
                    loss_D += 0.0 * y_real.squeeze()[0]
                    loss_D /= float(num_accumulation)

                    scalars["loss/D/gradient_penalty"].append(r1_penalty.detach())

                self.scaler_r1.scale(loss_D).backward()

            # update D parameters
            self.scaler_r1.step(self.optim_D)
            self.scaler_r1.update()

        set_requires_grad(self.D, False)

        #############################################################
        # exiting step
        #############################################################

        ema_imgs = int(self.cfg.training.ema_kimg * 1e3)
        if self.cfg.training.ema_rampup is not None:
            cur_imgs = iteration * self.cfg.training.batch_size
            ema_imgs = min(ema_imgs, cur_imgs * self.cfg.training.ema_rampup)
        ema_decay = 0.5 ** (self.cfg.training.batch_size / max(ema_imgs, 1e-8))
        ema_inplace(self.G_ema, self.G.module, ema_decay)

        if iteration % self.cfg.training.lazy.ada == 0:
            rt = self.A.update_p()
            scalars["stats/ada_rt"] = [rt.detach()]
            scalars["stats/ada_p"] = [self.A.p.detach()]

        # gather scalars from all devices
        for key, scalar_list in scalars.items():
            scalar = torch.mean(torch.stack(scalar_list))
            dist.all_reduce(scalar)  # sum over gpus
            scalar /= self.cfg.training.num_gpus
            scalars[key] = scalar.detach().cpu().item()

        scalars["stats/ema_decay"] = ema_decay
        scalars["stats/warmup_blur_sigma"] = self.blur_sigma
        scalars["stats/warmup_dropout_ratio"] = self.dropout_ratio

        return scalars

    @torch.no_grad()
    def sample(self, ema=False):
        if ema:
            self.G_ema.eval()
            fakes = self.G_ema(self.z_fixed, **self.auxin)
        else:
            self.G.eval()
            fakes = self.G(self.z_fixed, **self.auxin)
        return fakes

    @torch.no_grad()
    def validation(self):
        # only use feature-based metrics to save training time
        torch.cuda.empty_cache()
        N = 10_000
        B = self.cfg.training.batch_size_per_gpu
        B_list = [B] * (N // B) + ([N % B] if N % B != 0 else [])
        pointnet = pretrained_pointnet().to(self.device)
        self.G_ema.eval()

        def get_pointnet_features(depth):
            depth = tanh_to_sigmoid(depth).clamp(0, 1)
            points = self.coord.convert(depth, "inv_depth_norm", "point_set")
            points /= self.coord.max_depth
            feats = pointnet(points.transpose(1, 2))
            return feats

        tqdm_kwargs = dict(dynamic_ncols=True, leave=False)

        # real data (cached)
        if self.val_real_feats is None:
            train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.batch_size_per_gpu,
                num_workers=self.cfg.training.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
            self.val_real_feats = []
            for item in tqdm(train_loader, desc="real", **tqdm_kwargs):
                depth = self.fetch_reals(item)["image"]
                points = get_pointnet_features(depth).cpu()
                self.val_real_feats.append(points)
            self.val_real_feats = torch.cat(self.val_real_feats, dim=0)

        # fake data
        val_fake_feats = []
        for batch_size in tqdm(B_list, desc="fake", **tqdm_kwargs):
            z = self.sample_z(batch_size)
            angle = self.coord.angle.repeat_interleave(batch_size, dim=0)
            depth = self.G_ema(z, angle=angle)["image"]
            points = get_pointnet_features(depth).cpu()
            val_fake_feats.append(points)
        val_fake_feats = torch.cat(val_fake_feats, dim=0)

        scores = {}
        scores[f"pointcloud/frechet_distance_{N//1000}k"] = compute_frechet_distance(
            feats1=val_fake_feats.numpy(),
            feats2=self.val_real_feats.numpy(),
        )
        scores[f"pointcloud/squared_mmd_{N//1000}k"] = compute_squared_mmd(
            feats1=val_fake_feats.numpy(),
            feats2=self.val_real_feats.numpy(),
        )
        return scores

    def save_checkpoint(self, save_path, step):
        ckpt = {
            "cfg": self.cfg,
            "step": step,
            "angle": self.coord.angle.detach().cpu(),
            "G": self.G.module.state_dict(),
            "D": self.D.module.state_dict(),
            "G_ema": self.G_ema.state_dict(),
            "A": self.A.state_dict(),
            "optim_G": self.optim_G.state_dict(),
            "optim_D": self.optim_D.state_dict(),
        }
        if self.cfg.training.loss.pl > 0.0:
            ckpt["pl_ema"] = self.pl_ema.detach().cpu()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, save_path)
