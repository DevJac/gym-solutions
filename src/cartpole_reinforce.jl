# Inspired by: https://github.com/udacity/deep-reinforcement-learning/blob/master/reinforce/REINFORCE.ipynb
module CartPoleREINFORCE

using Flux
using OpenAIGym
using Printf
import Reinforce: action

const env = GymEnv(:CartPole, :v0)
const state_size = 4
const actions = 0:1

function make_model(hidden_layer_size=16)
    Chain(
        Dense(state_size, hidden_layer_size, relu),
        Dense(hidden_layer_size, length(actions), identity),
        softmax)
end

function loss(model, sars)
    -sum(
        map(sars) do sars
            sars.q * log(model(sars.s)[sars.a + 1])
        end
    )
end

const default_optimizer = AMSGrad()

function train!(model, sars, optimizer=default_optimizer)
    Flux.train!((sars) -> loss(model, sars), Flux.params(model), [(sars,)], optimizer)
end

struct Policy <: AbstractPolicy
    model
end

function action(policy::Policy, r, s, A)
    sample(actions, Weights(policy.model(s)))
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
    model = make_model()
    all_rewards = []
    for episode in 1:10_000
        sars, episode_rewards = run_episodes(1, Policy(model))
        append!(all_rewards, episode_rewards)
        recent_rewards = length(all_rewards) >= 100 ? all_rewards[end-99:end] : all_rewards
        train!(model, sars)
        if episode % 100 == 0
            @printf("Episode: %4d    Mean of recent rewards: %6.2f\n", episode, mean(recent_rewards))
        end
        if mean(recent_rewards) >= 195
            return episode, model
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
    CartPoleREINFORCE.run()
end
