module LunarLanderREINFORCE

using Flux
using LinearAlgebra
using OpenAIGym
using Printf
import Reinforce: action

const env = GymEnv(:LunarLander, :v2)
const state_size = length(env.state)
const actions = 0:3

function make_models(hidden_layer_size=64)
    policy = Chain(
        Dense(state_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(actions), identity),
        softmax)
    value = Chain(
        Dense(state_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        v -> v[1])
    (policy, value)
end

function p_loss(p_model, v_model, sars; entropy_bonus=0.1, regularization_pressure=0.01)
    -sum(
        map(sars) do sars
            p = p_model(sars.s)
            (sars.q - v_model(sars.s)) * log(p[sars.a + 1]) + entropy_bonus * entropy(p)
        end
    ) + regularization_pressure * sum(norm, Flux.params(p_model))
end

const default_p_optimizer = AMSGrad()

function p_train!(p_model, v_model, sars, optimizer=default_p_optimizer)
    Flux.train!((sars) -> p_loss(p_model, v_model, sars), Flux.params(p_model), [(sars,)], optimizer)
end

function v_loss(v_model, sars; regularization_pressure=10)
    sum(
        map(sars) do sars
            (sars.q - v_model(sars.s)) ^ 2
        end
    ) + regularization_pressure * sum(norm, Flux.params(v_model))
end

const default_v_optimizer = AMSGrad()

function v_train!(v_model, sars, optimizer=default_v_optimizer)
    Flux.train!((sars) -> v_loss(v_model, sars), Flux.params(v_model), [(sars,)], optimizer)
end

struct Policy <: AbstractPolicy
    p_model
end

function action(policy::Policy, r, s, A)
    sample(actions, Weights(policy.p_model(s)))
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
        single_episode_reward = run_episode(env, policy) do (s, a, r, s_next)
            push!(sars, SARS(s, a, r, nothing, s_next, finished(env)))
            if render_env; render(env) end
        end
        push!(rewards, single_episode_reward)
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

function reinforce()
    last_progress_output = 0
    p_model, v_model = make_models()
    all_rewards = []
    for episode in 1:10_000
        output_progress = time() > last_progress_output + 20
        sars, episode_rewards = run_episodes(1, Policy(p_model), render_env=output_progress)
        append!(all_rewards, episode_rewards)
        recent_rewards = length(all_rewards) >= 100 ? all_rewards[end-99:end] : all_rewards
        v_train!(v_model, sars)
        p_train!(p_model, v_model, sars)
        if output_progress
            @printf("Episode: %4d    Mean of recent rewards: %7.2f\n", episode, mean(recent_rewards))
            last_progress_output = time()
        end
        if episode >= 100 && mean(recent_rewards) >= 200
            return episode, p_model
        end
    end
end

function run()
    solved_in, solution = reinforce()
    @printf("Solved in %d episodes. Observe the solution.\n", solved_in)
    run_episodes(10, Policy(solution), render_env=true, close_env=true)
end

end # end module

if !isinteractive()
    LunarLanderREINFORCE.run()
end
