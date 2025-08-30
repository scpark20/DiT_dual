import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver

class DS_Solver(Solver):
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        algorithm_type="dual_prediction",
        checkpoint=False,
        **kwargs
    ):
        assert algorithm_type == 'dual_prediction'
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.flow_shift = flow_shift
        self.checkpoint = checkpoint
        
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))
        self.M = nn.Parameter(torch.zeros(steps, steps))
        
    def sample(self, x, model_fn, output_traj=False, **kwargs):
        self.set_model_fn(model_fn)
        
        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)  # <-- 학습된 ts
        # noise_schedule이 텐서를 받아들일 수 있어야 자동미분이 유지됩니다.
        alphas = self.noise_schedule.marginal_alpha(timesteps)          # 벡터화된 구현 권장
        sigmas = self.noise_schedule.marginal_std(timesteps)
        delta_alphas = alphas[1:] - alphas[:-1]
        delta_sigmas = sigmas[1:] - sigmas[:-1]
        
        vs = []
        trajs = [x]
        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            xc, ec = self.checkpoint_model_fn(x, timesteps[i]) if self.checkpoint else self.model_fn(x, timesteps[i])
            vc = delta_alphas[i]*xc + delta_sigmas[i]*ec
            if i == 0:
                x = x + vc
            else:
                # (i-1,)
                C = self.M[i, :i]
                if (self.steps <= 6) and i >= self.steps-2:
                    mask = torch.zeros_like(C); mask[-1] = 1.
                    C = C * mask
                    

                c_diag = 1 - torch.sum(C)
                V = torch.stack(vs, dim=0) # (N, b, c, h, w)
                x = x + c_diag*vc + torch.sum(C[:, None, None, None, None]*V, dim=0)
            vs.append(vc)
            if output_traj:
                trajs.append(x)    

        outputs = {'samples': x}
        if output_traj:
            outputs['traj'] = torch.stack(trajs, dim=1)
            outputs['timesteps'] = timesteps

        return outputs