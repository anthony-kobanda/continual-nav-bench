import torch

from typing import Dict



GOAL_SIZE = 3
ACTION_SIZE = 8

# OBS + (POS + GOAL) + (DELTA + DIR + DIST) + ANGLE
OBS_SIZE = 127 + (2 * GOAL_SIZE) + (2 * GOAL_SIZE + 1) + 2

HIGH_LAYER_ACT_CFG = {"classname":"torch.nn.GELU"}
HIGH_OUTPUT_ACT_CFG = {"classname":"torch.nn.Identity"}

LOW_LAYER_ACT_CFG = {"classname":"torch.nn.GELU"}
LOW_OUTPUT_ACT_CFG = {"classname":"torch.nn.Identity"}

USE_LAYER_NORM = True

MIN_DISTANCE = 2.0

# as we use tanh activation for both mean and log_std, we need to scale the output

HIGH_STD_FACTOR = 1.0

HIGH_LOG_STD_MAX = - 2.5
HIGH_LOG_STD_MIN = - 5.0

LOW_STD_FACTOR = 2.0

LOW_LOG_STD_MAX = 1.0
LOW_LOG_STD_MIN = - 2.0



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

    mean, log_std = outs["low_mean"][0], outs["low_log_std"][0].tanh()

    std = torch.clamp(LOW_STD_FACTOR * log_std, LOW_LOG_STD_MIN, LOW_LOG_STD_MAX).exp()

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

    x,y,z = list(outs["low_goal"].cpu().detach().numpy().flatten())

    return {
        "move_right": float(move[0].item()),
        "move_left": float(move[1].item()),
        "move_forwards": float(move[2].item()),
        "move_backwards": float(move[3].item()),
        "run": bool(run[0].item()),
        "jump": bool(jump[0].item()),
        "rotation": float(angle.item()),
        "subgx": float(x),
        "subgy": float(y),
        "subgz": float(z)
    }



def compute_loss(
        outs:Dict[str,torch.Tensor],
        batch:Dict[str, torch.Tensor],
        mask:torch.Tensor,
        log_infos:bool=False,
        compute_l1:bool=False,
        compute_l2:bool=False,
        **kwargs) -> Dict[str,torch.Tensor]:
    
    pos, low_goal, high_goal = batch["observation/pos"], batch["low_goal"], batch["high_goal"]
    
    # selection loss
    ################

    waypoint_selection_target = (torch.norm(pos - high_goal, dim=-1, keepdim=True) > MIN_DISTANCE).float()
    
    waypoint_selection_probs = outs["waypoint_selection_probs"]
    waypoint_selection_dist = torch.distributions.Bernoulli(waypoint_selection_probs)
    lp_selection = (waypoint_selection_dist.log_prob(waypoint_selection_target) * mask).sum(-1).sum(-1).mean()
    selection_loss = - lp_selection

    # high policy loss
    ##################

    high_target = low_goal - pos

    high_mean, high_log_std = outs["waypoint_mean"], outs["waypoint_log_std"].tanh()
    high_std = torch.clamp(HIGH_STD_FACTOR * high_log_std, HIGH_LOG_STD_MIN, HIGH_LOG_STD_MAX).exp()
    high_dist = torch.distributions.Normal(high_mean, high_std)
    lp_high:torch.Tensor = (high_dist.log_prob(high_target) * mask).sum(-1).sum(-1).mean()
    high_loss = - lp_high

    # low policy loss
    #################

    mean, log_std = outs["low_mean"], outs["low_log_std"].tanh()

    std = torch.clamp(LOW_STD_FACTOR * log_std, LOW_LOG_STD_MIN, LOW_LOG_STD_MAX).exp()

    move_prob = torch.sigmoid(mean[...,:4])
    run_prob = torch.sigmoid(mean[...,4:5])
    jump_prob = torch.sigmoid(mean[...,5:6])
    cos_sin_mean = mean[...,6:].tanh()
    cos_sin_std = std[...,6:]

    # move loss
    mr = batch["action/move_right"].float()
    ml = batch["action/move_left"].float()
    mf = batch["action/move_forwards"].float()
    mb = batch["action/move_backwards"].float()
    move_target = torch.cat([mr,ml,mf,mb],dim=-1)        
    move_dist = torch.distributions.Bernoulli(move_prob)
    lp_move = ((move_dist.log_prob(move_target)) * mask).sum(-1).sum(-1).mean()
    move_loss = - lp_move

    # run loss
    run_target = batch["action/run"].float()
    run_dist = torch.distributions.Bernoulli(run_prob)
    lp_run = ((run_dist.log_prob(run_target)) * mask).sum(-1).sum(-1).mean()
    run_loss = - lp_run

    # jump loss
    jump_target = batch["action/jump"].float()
    jump_dist = torch.distributions.Bernoulli(jump_prob)
    lp_jump = ((jump_dist.log_prob(jump_target)) * mask).sum(-1).sum(-1).mean()
    jump_loss = - lp_jump

    # rotation loss
    cos = batch["action/rotation"].cos()
    sin = batch["action/rotation"].sin()
    cos_sin_target = torch.cat([cos,sin],dim=-1)

    SIZES = cos_sin_mean.size()[:-1]
    PROD_SIZES = 1
    for s in SIZES:
        PROD_SIZES *= s
    
    cos_sin_mean = cos_sin_mean.reshape(PROD_SIZES,2)
    cos_sin_std = cos_sin_std.reshape(PROD_SIZES,2)
    cos_sin_target = cos_sin_target.reshape(PROD_SIZES,2)
    rot_dist = torch.distributions.Normal(cos_sin_mean,cos_sin_std)
    logp_pi:torch.Tensor = rot_dist.log_prob(cos_sin_target).sum(-1)
    logp_pi = ((logp_pi.reshape(*SIZES).unsqueeze(-1)) * mask).sum(-1).sum(-1).mean()
    rotation_loss = - logp_pi

    # total low loss
    low_loss = move_loss + run_loss + jump_loss + rotation_loss

    # l1 loss

    selection_l1_loss = torch.tensor(0.0)
    high_l1_loss = torch.tensor(0.0)

    move_l1_loss = torch.tensor(0.0)
    run_l1_loss = torch.tensor(0.0)
    jump_l1_loss = torch.tensor(0.0)
    rotation_l1_loss = torch.tensor(0.0)
    low_l1_loss = torch.tensor(0.0)

    if compute_l1:

        selection_predicted = waypoint_selection_dist.sample()
        selection_l1_loss = (torch.abs(selection_predicted - waypoint_selection_target) * mask).sum(-1).sum(-1).mean()

        high_predicted = high_dist.sample()
        high_l1_loss = (torch.abs(high_predicted - high_target) * mask).sum(-1).mean()

        move_predicted = move_dist.sample()
        run_predicted = run_dist.sample()
        jump_predicted = jump_dist.sample()
        cos_sin_predicted = rot_dist.sample().reshape(*SIZES,2)
        cos_sin_target = cos_sin_target.reshape(*SIZES,2)
        move_l1_loss = (torch.abs(move_predicted - move_target) * mask).sum(-1).sum(-1).mean()
        run_l1_loss = (torch.abs(run_predicted - run_target) * mask).sum(-1).sum(-1).mean()
        jump_l1_loss = (torch.abs(jump_predicted - jump_target) * mask).sum(-1).sum(-1).mean()
        rotation_l1_loss = (torch.abs(cos_sin_predicted - cos_sin_target) * mask).sum(-1).sum(-1).mean()
        low_l1_loss = move_l1_loss + run_l1_loss + jump_l1_loss + rotation_l1_loss  
    
    # l2 loss

    selection_l2_loss = torch.tensor(0.0)
    high_l2_loss = torch.tensor(0.0)

    move_l2_loss = torch.tensor(0.0)
    run_l2_loss = torch.tensor(0.0)
    jump_l2_loss = torch.tensor(0.0)
    rotation_l2_loss = torch.tensor(0.0)
    low_l2_loss = torch.tensor(0.0)

    if compute_l2:

        selection_predicted = waypoint_selection_dist.sample()
        selection_l2_loss = ((selection_predicted - waypoint_selection_target).pow(2) * mask).sum(-1).sum(-1).mean()

        high_predicted = high_dist.sample()
        high_l2_loss = ((high_predicted - high_target).pow(2) * mask).sum(-1).sum(-1).mean()
        
        move_predicted = move_dist.sample()
        run_predicted = run_dist.sample()
        jump_predicted = jump_dist.sample()
        cos_sin_predicted = rot_dist.sample().reshape(*SIZES,2)
        cos_sin_target = cos_sin_target.reshape(*SIZES,2)
        move_l2_loss = ((move_predicted - move_target).pow(2) * mask).sum(-1).sum(-1).mean()
        run_l2_loss = ((run_predicted - run_target).pow(2) * mask).sum(-1).sum(-1).mean()
        jump_l2_loss = ((jump_predicted - jump_target).pow(2) * mask).sum(-1).sum(-1).mean()
        rotation_l2_loss = ((cos_sin_predicted - cos_sin_target).pow(2) * mask).sum(-1).sum(-1).mean()
        low_l2_loss = move_l2_loss + run_l2_loss + jump_l2_loss + rotation_l2_loss

    if log_infos:

        # Mean STD
        ##########
        
        high_std = (high_std * mask).mean(-1).sum(-1).mean()

        cos_sin_std = cos_sin_std.reshape(*SIZES,2)
        rotation_std = (cos_sin_std * mask).mean(-1).sum(-1).mean()

        return {
            "A1(Main)_low_policy_loss": low_loss,
            "A1(Main)_high_policy_loss": high_loss + selection_loss,
            "A2(Infos)_high_selection_loss": selection_loss,

            "A2(Infos)_low_l1_loss": low_l1_loss,
            "A2(Infos)_high_l1_loss": high_l1_loss,
            "A2(Infos)_high_selection_l1_loss": selection_l1_loss,

            "A2(Infos)_low_l2_loss": low_l2_loss,
            "A2(Infos)_high_l2_loss": high_l2_loss,
            "A2(Infos)_high_selection_l2_loss": selection_l2_loss,

            "A2(Infos)_move_loss": move_loss,
            "A2(Infos)_run_loss": run_loss,
            "A2(Infos)_jump_loss": jump_loss,
            "A2(Infos)_rotation_loss": rotation_loss,
            "A2(Infos)_rotation_l2_loss": rotation_l2_loss,

            "A2(Infos)_move_l1_loss": move_l1_loss,
            "A2(Infos)_run_l1_loss": run_l1_loss,
            "A2(Infos)_jump_l1_loss": jump_l1_loss,
            "A2(Infos)_rotation_l1_loss": rotation_l1_loss,

            "A2(Infos)_move_l2_loss": move_l2_loss,
            "A2(Infos)_run_l2_loss": run_l2_loss,
            "A2(Infos)_jump_l2_loss": jump_l2_loss,
            "A2(Infos)_rotation_l2_loss": rotation_l2_loss,

            "A2(Infos)_high_std": high_std,
            "A2(Infos)_rotation_std": rotation_std,

        }

    return {
        "A1(Main)_low_policy_loss": low_loss,
        "A1(Main)_high_policy_loss": high_loss + selection_loss,
        "A2(Infos)_high_selection_loss": selection_loss,

        "A2(Infos)_low_l1_loss": low_l1_loss,
        "A2(Infos)_high_l1_loss": high_l1_loss,
        "A2(Infos)_high_selection_l1_loss": selection_l1_loss,

        "A2(Infos)_low_l2_loss": low_l2_loss,
        "A2(Infos)_high_l2_loss": high_l2_loss,
        "A2(Infos)_high_selection_l2_loss": selection_l2_loss,

        "A2(Infos)_move_loss": move_loss,
        "A2(Infos)_run_loss": run_loss,
        "A2(Infos)_jump_loss": jump_loss,
        "A2(Infos)_rotation_loss": rotation_loss,
        "A2(Infos)_rotation_l2_loss": rotation_l2_loss,

        "A2(Infos)_move_l1_loss": move_l1_loss,
        "A2(Infos)_run_l1_loss": run_l1_loss,
        "A2(Infos)_jump_l1_loss": jump_l1_loss,
        "A2(Infos)_rotation_l1_loss": rotation_l1_loss,

        "A2(Infos)_move_l2_loss": move_l2_loss,
        "A2(Infos)_run_l2_loss": run_l2_loss,
        "A2(Infos)_jump_l2_loss": jump_l2_loss,
        "A2(Infos)_rotation_l2_loss": rotation_l2_loss,
    }
