# ==========================================
# 3. 訓練工具與獎勵機制 (V.24 積分系數調整版)
# ==========================================
import os
import copy
import random
import numpy as np
import torch
import torch.optim as optim

def encode_obs(obs): return obs.astype(np.float32)

def select_action(agent, state):
    device = next(agent.parameters()).device
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = agent(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action).item(), value.item()

def random_action(): return random.randint(0, len(ACTIONS) - 1)

def compute_reward(result_dict, stage, points_this_round=0):
    """ 💥 V.24 積分賽制系數優化 """
    # 1. 租金獎勵：每回合對中一爻給 15 分 (加強地產意識)
    reward = points_this_round * 15.0 
    
    rank = result_dict.get("rank")
    status = result_dict.get("status")
    turns = result_dict.get("turn", 0)

    # 2. 破產重罰 (5代幣模式下破產是極低標錯誤)
    if rank == Rank.B: reward -= 1000.0

    # --- 根據階段給予不同終局權重 ---
    
    # Stage 1-2: 鼓勵達成目標與判定勝
    if stage <= 2:
        if rank in [Rank.S, Rank.A, Rank.C]: reward += 500.0
        if status == "DECISION_WIN": reward += 200.0
            
    # Stage 3-4: 引入速通紅利與重罰平局
    else:
        # 重罰平局與判定敗，強迫 AI 必須在 100 回合內分出勝負
        if status == "DECISION_WIN": reward += 500.0
        elif status == "DECISION_LOSS": reward -= 500.0
        elif status == "DRAW": reward -= 1000.0 # 💥 平局即地獄
            
        if rank in [Rank.S, Rank.A, Rank.C]:
            # 💥 積分馬拉松速通定義
            if turns <= 15:    # 神級速度 (平均每回合收租 > 4.2分)
                reward += 2500.0 
            elif turns <= 30:  # 優良速度
                reward += 1500.0
            elif turns <= 50:  # 普通完賽
                reward += 800.0
            else:              # 慢速完賽
                reward += 200.0 

    return reward

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv.insert(0, gae)
    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns

# ==========================================
# 4. 競技場主迴圈
# ==========================================
def train_ppo(env_class, episodes=24000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [V.24 積分爭霸] 5-Tokens | 64分制 | 23維觀察 | 裝置: {device}", flush=True)
    
    agent = PPOAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    opponent_pool = [copy.deepcopy(agent.state_dict())] 
    wisdom_buffer = [] 
    
    clip_eps, entropy_coef, value_coef = 0.2, 0.01, 0.5
    S_count, A_count, D_count, B_count = 0, 0, 0, 0 

    print("🌱 23維神經網路初始化完成。開始 4 階段課程演化...", flush=True)

    for ep in range(episodes):
        env = env_class()
        memory = []
        
        # 課程階段判定
        if ep < 6000: stage = 1; use_random_opponent = True
        elif ep < 12000: stage = 2; use_random_opponent = (random.random() < 0.5)
        elif ep < 18000: stage = 3; use_random_opponent = False
        else: stage = 4; use_random_opponent = False
            
        opp_model = None
        if not use_random_opponent:
            opp_model = PPOAgent().to(device)
            opp_model.load_state_dict(random.choice(opponent_pool))
            opp_model.eval()

        # Turn 1 (Silent Phase)
        obs_a, obs_b = encode_obs(env.observe("A")), encode_obs(env.observe("B"))
        a_act, a_logp, a_val = select_action(agent, obs_a)
        b_act = random_action() if use_random_opponent else select_action(opp_model, obs_b)[0]
        
        res_a, res_b = env.step_silent(ACTIONS[a_act], ACTIONS[b_act])
        
        # 結算第一回合收租
        env.resolve_round() 
        pts_earned_first = env.players["A"].total_points
        r_a = compute_reward(res_a, stage, pts_earned_first)
        
        done_a = not env.players["A"].active
        memory.append((obs_a, a_act, a_logp, a_val, r_a, done_a))

        # Open Phase (積分賽延長至 100 回合)
        max_turns = 100 
        
        while env.players["A"].active and env.players["B"].active:
            if env.turn > max_turns: 
                # 💥 判定勝負
                pts_a, pts_b = env.players["A"].total_points, env.players["B"].total_points
                if pts_a > pts_b: res_a["status"] = "DECISION_WIN"
                elif pts_b > pts_a: res_a["status"] = "DECISION_LOSS"
                else: res_a["status"] = "DRAW"
                    
                final_reward = compute_reward(res_a, stage, 0)
                last_exp = memory[-1]
                memory[-1] = (last_exp[0], last_exp[1], last_exp[2], last_exp[3], last_exp[4] + final_reward, True)
                break
            
            player = next(env.ab_ba_seq)
            obs = encode_obs(env.observe(player))
            
            if player == "A":
                act, logp, val = select_action(agent, obs)
                result = env.step_sequential(player, ACTIONS[act])
                
                pts_earned = 0
                if result["round_complete"]: 
                    pts_before = env.players["A"].total_points
                    env.resolve_round()
                    pts_earned = env.players["A"].total_points - pts_before

                reward = compute_reward(result, stage, pts_earned)
                memory.append((obs, act, logp, val, reward, not env.players["A"].active))
                res_a = result 
            else:
                act = random_action() if use_random_opponent else select_action(opp_model, obs)[0]
                result = env.step_sequential(player, ACTIONS[act])
                if result["round_complete"]: env.resolve_round()

        # 結算數據
        status = res_a["status"]
        if status == "RANK_B_COLLAPSE": B_count += 1
        elif status == "GOAL_REACHED" or res_a.get("rank") in [Rank.S, Rank.A]:
            if res_a.get("rank") == Rank.S: S_count += 1
            else: A_count += 1
            wisdom_buffer.append(memory)
        elif status == "DECISION_WIN":
            D_count += 1
            if stage >= 2: wisdom_buffer.append(memory)
        
        if len(wisdom_buffer) > 60: wisdom_buffer.pop(0)

        # PPO 更新公式
        update_batch = list(memory)
        if len(wisdom_buffer) > 0 and random.random() < 0.2:
            update_batch += random.choice(wisdom_buffer)

        if len(update_batch) > 1:
            states, actions, logps, values, rewards, dones = zip(*update_batch)
            advs, returns = compute_gae(list(rewards), list(values), list(dones))
            states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions_t = torch.tensor(actions).to(device)
            old_logps_t = torch.tensor(logps).to(device)
            returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
            advs_t = torch.tensor(advs, dtype=torch.float32).to(device)
            for _ in range(4):
                logits, value = agent(states_t)
                dist = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
                new_logps = dist.log_prob(actions_t)
                ratio = torch.exp(new_logps - old_logps_t)
                surr1 = ratio * advs_t
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advs_t
                loss = -torch.min(surr1, surr2).mean() + value_coef * (returns_t - value.squeeze()).pow(2).mean() - entropy_coef * dist.entropy().mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Log 輸出
        if (ep + 1) % 50 == 0:
            phase_str = "vs Random" if use_random_opponent else "vs Pool"
            wr = ((S_count + A_count + D_count) / 50) * 100
            print(f"🔹 Stage {stage} | Ep {ep+1:5d} [{phase_str}] | S: {S_count:2d} | A: {A_count:2d} | D: {D_count:2d} | B: {B_count:2d} | WR: {wr:.1f}%", flush=True)
            S_count, A_count, D_count, B_count = 0, 0, 0, 0
            
        if (ep + 1) % 1000 == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))

    torch.save(agent.state_dict(), "ppo_six_ngaau_v24.pth")
    print("🏁 V.24 積分爭霸演化結束。模型已存檔。")

if __name__ == "__main__":
    train_ppo(MultiAgentSixNgaauEnv, episodes=24000)
