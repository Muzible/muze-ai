"""
🚀 Latent Consistency Model (LCM) Distillation

Distylacja z pełnego modelu dyfuzji (200 kroków) do modelu spójności latentnej (4-8 kroków).

Bazowane na:
- "Latent Consistency Models" (Luo et al., 2023)
- "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module" (Luo et al., 2023)

Główne idee:
1. Model LCM uczy się przewidywać końcowy wynik zamiast szumu
2. Self-consistency: LCM(z_t1, t1) ≈ LCM(z_t2, t2) dla dowolnych t1, t2
3. Skip timesteps: trening na przeskokach (np. 200→160→120→80→40→0)
4. Distylacja: uczymy się z nauczyciela (pełny LDM) przez consistency loss

Pipeline:
    Teacher LDM (200 kroków, zamrożony)
         ↓
    LCM Student (4-8 kroków)
         ↓
    Consistency Loss + Distillation Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from copy import deepcopy
import math


class LCMScheduler:
    """
    Scheduler dla Latent Consistency Model.
    
    Kluczowe różnice od DDPM scheduler:
    - Mniej kroków (4-8 zamiast 200-1000)
    - Skipping timesteps z równomiernym rozkładem
    - Guidance scale wbudowany w model
    """
    
    def __init__(
        self,
        num_inference_steps: int = 4,
        original_steps: int = 200,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        self.num_inference_steps = num_inference_steps
        self.original_steps = original_steps
        
        # Original noise schedule
        if beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, original_steps) ** 2
        else:
            betas = torch.linspace(beta_start, beta_end, original_steps)
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.alphas_cumprod = alphas_cumprod
        
        # Compute timesteps for LCM (równomiernie rozłożone)
        self.timesteps = self._get_timesteps()
    
    def _get_timesteps(self) -> torch.Tensor:
        """Oblicza timesteps dla LCM inferencji"""
        # Równomiernie rozłożone timesteps
        # np. dla 4 kroków: [199, 149, 99, 49, 0] lub podobnie
        step_ratio = self.original_steps // self.num_inference_steps
        timesteps = torch.arange(self.num_inference_steps) * step_ratio
        timesteps = self.original_steps - 1 - timesteps
        return timesteps.flip(0)  # Od niskiego do wysokiego noise level
    
    def get_scalings(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zwraca skalowanie dla danego timestep"""
        alpha_prod_t = self.alphas_cumprod[t]
        sigma_t = torch.sqrt(1 - alpha_prod_t)
        alpha_t = torch.sqrt(alpha_prod_t)
        return alpha_t, sigma_t


class LCMDistillationTrainer:
    """
    Trener do distylacji LCM z pełnego modelu LDM.
    
    Proces:
    1. Zamrażamy teacher LDM
    2. Inicjalizujemy student jako kopię teachera
    3. Trenujemy studenta na consistency loss
    
    Consistency Loss:
    - Student powinien przewidywać ten sam x_0 niezależnie od t
    - LCM(z_t1, t1) ≈ LCM(z_t2, t2) ≈ x_0
    
    Distillation Loss:
    - Student powinien naśladować wyjście teachera
    - |Student(z_t, t) - Teacher_ODE_solve(z_t, t)|^2
    """
    
    def __init__(
        self,
        teacher_ldm: nn.Module,
        student_unet: nn.Module = None,
        num_train_timesteps: int = 200,
        num_inference_steps: int = 4,
        w_min: float = 3.0,  # Min guidance scale dla CFG
        w_max: float = 15.0,  # Max guidance scale dla CFG
        device: str = 'cuda',
    ):
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.w_min = w_min
        self.w_max = w_max
        
        # Teacher (zamrożony)
        self.teacher = teacher_ldm
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Student (kopia teachera lub podany)
        if student_unet is None:
            self.student_unet = deepcopy(teacher_ldm.unet)
        else:
            self.student_unet = student_unet
        self.student_unet.train()
        
        # Noise schedule
        self.scheduler = LCMScheduler(
            num_inference_steps=num_inference_steps,
            original_steps=num_train_timesteps,
        )
        
        # EMA dla studenta (stabilniejszy trening)
        self.ema_student = None
        self.ema_decay = 0.9999
        
    def init_ema(self):
        """Inicjalizuje EMA studenta"""
        self.ema_student = deepcopy(self.student_unet)
        self.ema_student.eval()
        for param in self.ema_student.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_ema(self):
        """Aktualizuje EMA studenta"""
        if self.ema_student is None:
            return
        
        for ema_param, param in zip(self.ema_student.parameters(), self.student_unet.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def get_w(self) -> torch.Tensor:
        """Losuje guidance scale z rozkładu log-uniform"""
        # Log-uniform distribution między w_min i w_max
        log_w = torch.rand(1, device=self.device) * (math.log(self.w_max) - math.log(self.w_min)) + math.log(self.w_min)
        return torch.exp(log_w)
    
    def predicted_x0(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Oblicza przewidywany x_0 z outputu modelu (epsilon) i noisy sample"""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        alpha_t = torch.sqrt(alphas_cumprod[timestep])[:, None, None, None]
        sigma_t = torch.sqrt(1 - alphas_cumprod[timestep])[:, None, None, None]
        
        # x_0 = (z_t - sigma_t * eps) / alpha_t
        pred_x0 = (sample - sigma_t * model_output) / alpha_t.clamp(min=1e-8)
        return pred_x0
    
    @torch.no_grad()
    def teacher_solve_ode(
        self,
        z_t: torch.Tensor,
        t_start: int,
        t_end: int,
        text_embed: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Rozwiązuje ODE od t_start do t_end używając teachera.
        
        Używa DDIM-like update dla efektywności.
        """
        z = z_t.clone()
        
        # Interpoluj timesteps między t_start i t_end
        num_steps = max(1, abs(t_start - t_end) // 20)  # ~20 kroków na segment
        timesteps = torch.linspace(t_start, t_end, num_steps + 1, dtype=torch.long, device=self.device)
        
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_batch = torch.full((z.shape[0],), t_cur, device=self.device, dtype=torch.long)
            
            # Predykcja szumu
            eps = self.teacher.unet(z, t_batch, text_embed, **kwargs)
            
            # DDIM update
            alpha_cur = alphas_cumprod[t_cur]
            alpha_next = alphas_cumprod[t_next]
            
            # Predicted x0
            pred_x0 = (z - torch.sqrt(1 - alpha_cur) * eps) / torch.sqrt(alpha_cur).clamp(min=1e-8)
            
            # Clamp for stability
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Next sample
            z = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps
        
        return z
    
    def consistency_loss(
        self,
        z_0: torch.Tensor,
        text_embed: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Oblicza consistency loss.
        
        LCM powinien przewidywać ten sam x_0 dla różnych poziomów szumu.
        """
        B = z_0.shape[0]
        device = z_0.device
        
        # Losuj guidance scale
        w = self.get_w()
        
        # Losuj parę timesteps (t_n, t_n+k) gdzie k to skip
        # Używamy timesteps ze schedulera
        idx = torch.randint(0, self.num_inference_steps - 1, (B,), device=device)
        t_n = self.scheduler.timesteps[idx].to(device)
        t_n_plus_k = self.scheduler.timesteps[idx + 1].to(device)
        
        # Dodaj szum do z_0 dla obu timesteps
        noise = torch.randn_like(z_0)
        
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        
        # z_{t_n} = sqrt(alpha_n) * z_0 + sqrt(1-alpha_n) * noise
        alpha_n = alphas_cumprod[t_n][:, None, None, None]
        z_t_n = torch.sqrt(alpha_n) * z_0 + torch.sqrt(1 - alpha_n) * noise
        
        # z_{t_n+k} = sqrt(alpha_{n+k}) * z_0 + sqrt(1-alpha_{n+k}) * noise
        alpha_n_k = alphas_cumprod[t_n_plus_k][:, None, None, None]
        z_t_n_k = torch.sqrt(alpha_n_k) * z_0 + torch.sqrt(1 - alpha_n_k) * noise
        
        # Student predictions
        # Dla z_t_n przewidujemy x_0
        eps_n = self.student_unet(z_t_n, t_n, text_embed, **kwargs)
        pred_x0_n = self.predicted_x0(eps_n, t_n, z_t_n)
        
        # Dla z_t_n+k teacher rozwiązuje ODE do t_n, potem student przewiduje
        with torch.no_grad():
            # Teacher: z_{t_n+k} → z_{t_n} (przez ODE)
            z_teacher_at_n = self.teacher_solve_ode(
                z_t_n_k, 
                int(t_n_plus_k[0].item()), 
                int(t_n[0].item()),
                text_embed,
                **kwargs
            )
        
        # Student na z_teacher_at_n
        eps_n_target = self.student_unet(z_teacher_at_n.detach(), t_n, text_embed, **kwargs)
        pred_x0_n_target = self.predicted_x0(eps_n_target, t_n, z_teacher_at_n)
        
        # Consistency loss: predictions powinny być takie same
        # Używamy target z EMA studenta dla stabilności (jak w LCM paper)
        if self.ema_student is not None:
            with torch.no_grad():
                eps_ema = self.ema_student(z_teacher_at_n, t_n, text_embed, **kwargs)
                pred_x0_ema = self.predicted_x0(eps_ema, t_n, z_teacher_at_n)
            target = pred_x0_ema.detach()
        else:
            target = pred_x0_n_target.detach()
        
        # Huber loss (bardziej stabilny niż MSE)
        loss = F.huber_loss(pred_x0_n, target, delta=1.0)
        
        return {
            'consistency_loss': loss,
            'pred_x0_n': pred_x0_n,
            'target': target,
            'guidance_scale': w,
        }
    
    def distillation_loss(
        self,
        z_0: torch.Tensor,
        text_embed: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Distillation loss - student naśladuje rozwiązanie ODE teachera.
        
        Dla każdego z_t, student powinien przewidywać to samo co
        teacher po rozwiązaniu ODE do t=0.
        """
        B = z_0.shape[0]
        device = z_0.device
        
        # Losuj timestep
        t = torch.randint(1, self.num_train_timesteps, (B,), device=device)
        
        # Dodaj szum
        noise = torch.randn_like(z_0)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        alpha_t = alphas_cumprod[t][:, None, None, None]
        z_t = torch.sqrt(alpha_t) * z_0 + torch.sqrt(1 - alpha_t) * noise
        
        # Teacher: rozwiąż ODE z_t → z_0
        with torch.no_grad():
            teacher_pred = self.teacher_solve_ode(z_t, int(t[0].item()), 0, text_embed, **kwargs)
        
        # Student: bezpośrednia predykcja x_0
        eps_student = self.student_unet(z_t, t, text_embed, **kwargs)
        student_pred = self.predicted_x0(eps_student, t, z_t)
        
        # MSE loss
        loss = F.mse_loss(student_pred, teacher_pred.detach())
        
        return {
            'distillation_loss': loss,
            'student_pred': student_pred,
            'teacher_pred': teacher_pred,
        }
    
    def train_step(
        self,
        z_0: torch.Tensor,
        text_embed: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        consistency_weight: float = 1.0,
        distillation_weight: float = 0.5,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Pojedynczy krok treningowy.
        
        Args:
            z_0: Clean latent [B, C, H, W]
            text_embed: Text conditioning [B, seq, dim]
            optimizer: Optimizer dla studenta
            consistency_weight: Waga consistency loss
            distillation_weight: Waga distillation loss
            
        Returns:
            Dict ze stratami
        """
        optimizer.zero_grad()
        
        # Consistency loss
        c_out = self.consistency_loss(z_0, text_embed, **kwargs)
        
        # Distillation loss (opcjonalnie)
        d_out = self.distillation_loss(z_0, text_embed, **kwargs)
        
        # Total loss
        total_loss = (
            consistency_weight * c_out['consistency_loss'] + 
            distillation_weight * d_out['distillation_loss']
        )
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student_unet.parameters(), 1.0)
        
        optimizer.step()
        
        # Update EMA
        self.update_ema()
        
        return {
            'total_loss': total_loss.item(),
            'consistency_loss': c_out['consistency_loss'].item(),
            'distillation_loss': d_out['distillation_loss'].item(),
        }


class LatentConsistencyModel(nn.Module):
    """
    Latent Consistency Model - szybka inferencja w 4-8 krokach.
    
    Po distylacji, ten model może generować w 4-8 krokach
    zamiast 200 kroków pełnego LDM.
    
    Użycie:
        lcm = LatentConsistencyModel(trained_unet, num_steps=4)
        
        # Generacja
        z_0 = lcm.sample(
            shape=(1, 128, 8, 64),
            text_embed=text_embed,
            section_type=['chorus'],
        )
    """
    
    def __init__(
        self,
        unet: nn.Module,
        num_inference_steps: int = 4,
        original_steps: int = 200,
        guidance_scale: float = 7.5,
    ):
        super().__init__()
        
        self.unet = unet
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        self.scheduler = LCMScheduler(
            num_inference_steps=num_inference_steps,
            original_steps=original_steps,
        )
    
    def predicted_x0(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Oblicza przewidywany x_0"""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(sample.device)
        
        # Handle both scalar and batch timesteps
        if timestep.dim() == 0:
            alpha_t = torch.sqrt(alphas_cumprod[timestep])
            sigma_t = torch.sqrt(1 - alphas_cumprod[timestep])
        else:
            alpha_t = torch.sqrt(alphas_cumprod[timestep])[:, None, None, None]
            sigma_t = torch.sqrt(1 - alphas_cumprod[timestep])[:, None, None, None]
        
        pred_x0 = (sample - sigma_t * model_output) / alpha_t.clamp(min=1e-8)
        return pred_x0
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        text_embed: torch.Tensor,
        text_embed_uncond: torch.Tensor = None,
        section_type: Optional[List[str]] = None,
        position: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        voice_emb: Optional[torch.Tensor] = None,
        context_latent: Optional[torch.Tensor] = None,
        guidance_scale: float = None,
    ) -> torch.Tensor:
        """
        Generuje sample w num_inference_steps krokach.
        
        Args:
            shape: Kształt outputu (B, C, H, W)
            text_embed: Conditioning [B, seq, dim]
            text_embed_uncond: Unconditional embedding dla CFG
            guidance_scale: Override dla self.guidance_scale
            
        Returns:
            Generated latent z_0
        """
        device = text_embed.device
        B = shape[0]
        
        guidance_scale = guidance_scale or self.guidance_scale
        
        # Start from noise
        z = torch.randn(shape, device=device)
        
        timesteps = self.scheduler.timesteps.to(device)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        
        # LCM sampling loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise (opcjonalnie z CFG)
            if text_embed_uncond is not None and guidance_scale > 1.0:
                # CFG: eps = eps_uncond + w * (eps_cond - eps_uncond)
                eps_cond = self.unet(
                    z, t_batch, text_embed,
                    section_type=section_type,
                    position=position,
                    energy=energy,
                    tempo=tempo,
                    voice_emb=voice_emb,
                    context_latent=context_latent,
                )
                eps_uncond = self.unet(
                    z, t_batch, text_embed_uncond,
                    section_type=section_type,
                    position=position,
                    energy=energy,
                    tempo=tempo,
                    voice_emb=voice_emb,
                    context_latent=context_latent,
                )
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.unet(
                    z, t_batch, text_embed,
                    section_type=section_type,
                    position=position,
                    energy=energy,
                    tempo=tempo,
                    voice_emb=voice_emb,
                    context_latent=context_latent,
                )
            
            # Predicted x_0
            pred_x0 = self.predicted_x0(eps, t_batch, z)
            
            # Clamp for stability
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Next step (jeśli nie ostatni)
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = alphas_cumprod[t_next]
                
                # Add noise for next step
                # z_{t_next} = sqrt(alpha_next) * pred_x0 + sqrt(1-alpha_next) * eps
                z = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps
            else:
                # Last step - return pred_x0
                z = pred_x0
        
        return z
    
    def forward(
        self,
        text_embed: torch.Tensor,
        shape: Tuple[int, ...] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward = sample"""
        if shape is None:
            shape = (text_embed.shape[0], 128, 8, 64)  # Default shape
        return self.sample(shape, text_embed, **kwargs)


def train_lcm(
    teacher_ldm: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    vae: nn.Module,
    text_encoder: nn.Module,
    num_epochs: int = 10,
    num_inference_steps: int = 4,
    lr: float = 1e-5,
    device: str = 'cuda',
    checkpoint_dir: str = './checkpoints_v2',
    save_every: int = 2,
) -> LatentConsistencyModel:
    """
    Główna funkcja do treningu LCM.
    
    Args:
        teacher_ldm: Wytrenowany LatentDiffusionV2
        train_dataloader: DataLoader z danymi
        vae: AudioVAE do enkodowania audio
        text_encoder: EnhancedMusicEncoder do enkodowania tekstu
        num_epochs: Liczba epok
        num_inference_steps: Docelowa liczba kroków LCM
        lr: Learning rate
        device: Device
        checkpoint_dir: Gdzie zapisywać checkpointy
        save_every: Co ile epok zapisywać
        
    Returns:
        Wytrenowany LatentConsistencyModel
    """
    from pathlib import Path
    from tqdm import tqdm
    
    print("="*60)
    print("🚀 Training Latent Consistency Model (Phase 4)")
    print("="*60)
    print(f"   Inference steps: {num_inference_steps}")
    print(f"   Teacher timesteps: {teacher_ldm.num_timesteps}")
    print(f"   Learning rate: {lr}")
    
    # Trainer
    trainer = LCMDistillationTrainer(
        teacher_ldm=teacher_ldm,
        num_train_timesteps=teacher_ldm.num_timesteps,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    trainer.init_ema()
    
    # Move to device
    trainer.student_unet.to(device)
    if trainer.ema_student is not None:
        trainer.ema_student.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(trainer.student_unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Checkpoint dir
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_consistency = 0
        total_distillation = 0
        
        pbar = tqdm(train_dataloader, desc=f"LCM Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            audio = batch['audio'].to(device)
            prompts = batch.get('prompt', [''] * audio.shape[0])
            
            # Encode audio to latent
            with torch.no_grad():
                vae_out = vae(audio)
                z_0 = vae_out['z']
                
                # Encode text
                text_embed = text_encoder.encode_text_only(prompts)
                text_embed = text_embed.to(device)
            
            # Get section info if available
            kwargs = {}
            if 'section_type' in batch:
                kwargs['section_type'] = batch['section_type']
            if 'position' in batch:
                kwargs['position'] = batch['position'].to(device)
            if 'energy' in batch:
                kwargs['energy'] = batch['energy'].to(device)
            if 'tempo' in batch:
                kwargs['tempo'] = batch['tempo'].to(device)
            
            # Train step
            losses = trainer.train_step(z_0, text_embed, optimizer, **kwargs)
            
            total_loss += losses['total_loss']
            total_consistency += losses['consistency_loss']
            total_distillation += losses['distillation_loss']
            
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'cons': f"{losses['consistency_loss']:.4f}",
            })
        
        scheduler.step()
        
        # Epoch stats
        n_batches = len(train_dataloader)
        avg_loss = total_loss / n_batches
        avg_cons = total_consistency / n_batches
        avg_dist = total_distillation / n_batches
        
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f} (consistency: {avg_cons:.4f}, distillation: {avg_dist:.4f})")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'student_state_dict': trainer.student_unet.state_dict(),
                'ema_state_dict': trainer.ema_student.state_dict() if trainer.ema_student else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'num_inference_steps': num_inference_steps,
            }, checkpoint_dir / 'lcm_best.pt')
            print(f"   ✅ Saved best LCM (loss: {best_loss:.4f})")
        
        # Regular checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'student_state_dict': trainer.student_unet.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / f'lcm_epoch_{epoch+1}.pt')
    
    # Create final LCM
    lcm = LatentConsistencyModel(
        unet=trainer.ema_student if trainer.ema_student else trainer.student_unet,
        num_inference_steps=num_inference_steps,
        original_steps=teacher_ldm.num_timesteps,
    )
    
    print(f"\n✅ LCM training complete! Best loss: {best_loss:.4f}")
    print(f"   Model can generate in {num_inference_steps} steps (vs {teacher_ldm.num_timesteps} for teacher)")
    
    return lcm


if __name__ == "__main__":
    # Test
    print("Testing LCM components...")
    
    # Test scheduler
    scheduler = LCMScheduler(num_inference_steps=4, original_steps=200)
    print(f"Timesteps: {scheduler.timesteps}")
    
    print("✅ LCM module ready!")
