import torch

from typing import Dict



GOAL_SIZE = 3
ACTION_SIZE = 8

# OBS + (POS + GOAL) + (DELTA + DIR + DIST) + ANGLE
OBS_SIZE = 127 + (2 * GOAL_SIZE) + (2 * GOAL_SIZE + 1) + 2

LAYER_ACT_CFG = {"classname":"torch.nn.GELU"}
OUTPUT_ACT_CFG = {"classname":"torch.nn.Identity"}

USE_LAYER_NORM = True

# as we use tanh activation for both mean and log_std, we need to scale the output

STD_FACTOR = 2.0

LOG_STD_MAX = 1.0
LOG_STD_MIN = - 2.0



@torch.no_grad()
def additional_features(pos:torch.Tensor,goal:torch.Tensor) -> Dict[str,torch.Tensor]:

    results: Dict[str,torch.Tensor] = {}
    
    # distance and direction (normalized)
    results["delta"] = goal - pos
    results["distance"] = torch.norm(results["delta"],dim=-1,keepdim=True)
    results["direction"] = results["delta"] / (results["distance"] + 1e-6)

    # angle (in R^3)
    dx, dy, dz = results["direction"].chunk(3,dim=-1)
    azimuth = torch.atan2(dy,dx)
    elevation = torch.atan2(dz,torch.sqrt(dx.pow(2) + dy.pow(2)))
    results["angle"] = torch.cat([azimuth,elevation],dim=-1)

    return results



@torch.no_grad()
def forward(outs:Dict[str,torch.Tensor],stochastic:bool,generator:torch.Generator,**kwargs) -> Dict[str,torch.Tensor]:

    mean, log_std = outs["mean"][0], outs["log_std"][0].tanh()

    std = torch.clamp(STD_FACTOR * log_std, LOG_STD_MIN, LOG_STD_MAX).exp()

    move_prob = torch.sigmoid(mean[:4])
    run_prob = torch.sigmoid(mean[4:5])
    jump_prob = torch.sigmoid(mean[5:6])
    cos_sin_mean = mean[6:].tanh()
    cos_sin_std = std[6:]

    if stochastic:

        move = torch.bernoulli(move_prob,generator=generator)
        run = torch.bernoulli(run_prob,generator=generator)
        jump = torch.bernoulli(jump_prob,generator=generator)
        cos_sin = torch.normal(cos_sin_mean,cos_sin_std,generator=generator)
    
    else:

        move = move_prob.round()
        run = run_prob.round()
        jump = jump_prob.round()
        cos_sin = cos_sin_mean
    
    angle = torch.atan2(cos_sin[1],cos_sin[0])

    return {
        "move_right": float(move[0].item()),
        "move_left": float(move[1].item()),
        "move_forwards": float(move[2].item()),
        "move_backwards": float(move[3].item()),
        "run": bool(run[0].item()),
        "jump": bool(jump[0].item()),
        "rotation": float(angle.item())
    }



def compute_loss(
        outs:Dict[str,torch.Tensor],
        batch:Dict[str, torch.Tensor],
        mask:torch.Tensor,
        log_infos:bool=False,
        **kwargs) -> Dict[str,torch.Tensor]:

    mean, log_std = outs["mean"], outs["log_std"].tanh()

    std = torch.clamp(STD_FACTOR * log_std, LOG_STD_MIN, LOG_STD_MAX).exp()

    move_prob = torch.sigmoid(mean[:,:,:4])
    run_prob = torch.sigmoid(mean[:,:,4:5])
    jump_prob = torch.sigmoid(mean[:,:,5:6])
    cos_sin_mean = mean[:,:,6:].tanh()
    cos_sin_std = std[:,:,6:]

    # move loss
    mr = batch["action/move_right"].float()
    ml = batch["action/move_left"].float()
    mf = batch["action/move_forwards"].float()
    mb = batch["action/move_backwards"].float()
    move_target = torch.cat([mr,ml,mf,mb],dim=-1)        
    dist = torch.distributions.Bernoulli(move_prob)
    lp_move = ((dist.log_prob(move_target)) * mask).sum(-1).sum(-1).mean()
    move_loss = - lp_move

    # run loss
    run_target = batch["action/run"].float()
    dist = torch.distributions.Bernoulli(run_prob)
    lp_run = ((dist.log_prob(run_target)) * mask).sum(-1).sum(-1).mean()
    run_loss = - lp_run

    # jump loss
    jump_target = batch["action/jump"].float()
    dist = torch.distributions.Bernoulli(jump_prob)
    lp_jump = ((dist.log_prob(jump_target)) * mask).sum(-1).sum(-1).mean()
    jump_loss = - lp_jump

    # rotation loss
    cos = batch["action/rotation"].cos()
    sin = batch["action/rotation"].sin()
    cos_sin_target = torch.cat([cos,sin],dim=-1)
    B,T = cos_sin_mean.size()[:2]
    cos_sin_mean = cos_sin_mean.reshape(B*T,2)
    cos_sin_std = cos_sin_std.reshape(B*T,2)
    cos_sin_target = cos_sin_target.reshape(B*T,2)
    dist = torch.distributions.Normal(cos_sin_mean,cos_sin_std)
    logp_pi:torch.Tensor = dist.log_prob(cos_sin_target).sum(-1)
    logp_pi = ((logp_pi.reshape(B,T).unsqueeze(-1)) * mask).sum(-1).sum(-1).mean()
    rotation_loss = - logp_pi

    # total loss
    loss = move_loss + run_loss + jump_loss + rotation_loss

    if log_infos:

        cos_sin_predicted = dist.sample().reshape(B,T,2)
        cos_sin_target = cos_sin_target.reshape(B,T,2)

        rotation_l2_loss = ((cos_sin_predicted - cos_sin_target).pow(2) * mask).sum(-1).sum(-1).mean()

        # Mean STD
        ##########

        cos_sin_std = cos_sin_std.reshape(B,T,2)

        rotation_std = (cos_sin_std * mask).mean(-1).sum(-1).mean()

        return {
            "A1(Main)_policy_loss": loss,
            "A2(Infos)_move_loss": move_loss,
            "A2(Infos)_run_loss": run_loss,
            "A2(Infos)_jump_loss": jump_loss,
            "A2(Infos)_rotation_loss": rotation_loss,
            "A2(Infos)_rotation_std": rotation_std,
            "A2(Infos)_rotation_l2_loss": rotation_l2_loss
        }

    return {"A1(Main)_policy_loss": loss}
