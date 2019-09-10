# Based on: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
using Flux
using OpenAIGym
using Printf
using Statistics
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)
const state_size = length(env.state)
const actions = 0:3

function make_policy_network(hidden_layer_size=64)
    Chain(
        Dense(state_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(actions), identity),
        softmax)
end

function policy_network_loss(policy, sars)
    n_episodes = length(filter(sars -> sars.f, sars))
    -sum(
        map(sars) do sars
            # The training process will seek to maximize the following formula.
            # The policy network outputs a probability distribution over all actions.
            # The log of a probability is a negative number. Smaller probabilities are exponentially more negative.
            # Thus, to maximize the formula, the policy should output high probabilities for actions with high rewards.
            log(policy.policy_network(sars.s)[sars.a + 1]) * sars.q
        end
    ) / n_episodes
end

const policy_optimizer = AMSGrad()

function train_policy!(policy, sars)
    Flux.train!((sars) -> policy_network_loss(policy, sars), Flux.params(policy.policy_network), [(sars,)], policy_optimizer)
end

struct Policy <: AbstractPolicy
    policy_network
end

function action(policy::Policy, r, s, A)
    sample(actions, Weights(policy.policy_network(s)))
end

mutable struct SARS
    s
    a
    r
    q
    s_next
    f
end

function run_episodes(n_episodes, policy; render_env=false, close_env=false)
    sars = []
    rewards = []
    for episode in 1:n_episodes
        total_episode_reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(sars, SARS(s, a, r, nothing, s_next, finished(env)))
            if render_env && episode == n_episodes; render(env) end
        end
        push!(rewards, total_episode_reward)
    end
    if close_env; close(env) end
    fill_q!(sars)
    sars, rewards
end

function fill_q!(sars; discount_factor=1)
    q = 0
    for i in length(sars):-1:1
        q *= discount_factor
        if sars[i].f
            q = 0
        end
        q += sars[i].r
        sars[i].q = q
    end
end

function run()
    best_mean_reward = -Inf
    batch_size = 2
    total_episodes = 0
    policy = Policy(make_policy_network())
    for iteration in Iterators.countfrom(1)
        sars, rewards = run_episodes(batch_size, policy, render_env=true)
        total_episodes += batch_size
        if mean(rewards) > best_mean_reward
            best_mean_reward = mean(rewards)
            batch_size = max(2, batch_size - 10)
        else
            batch_size += 1
        end
        @printf("%4d: Batch Size: %4d  Reward: %7.2f (±%7.2f)  Best: %7.2f  Episodes: %5d\n",
                iteration, batch_size, mean(rewards), std(rewards), best_mean_reward, total_episodes)
        train_policy!(policy, sars)
    end
end
