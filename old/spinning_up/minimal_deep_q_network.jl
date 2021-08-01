module DQN

using DataStructures
using Flux
using OpenAIGym

Flux.@nograd onehot(x, set::DiscreteSet) = Flux.onehot(x, set.items)

struct QNetwork
    env
    network
end
Flux.@treelike QNetwork
(q::QNetwork)(s, a) = q.network(vcat(s, onehot(a, q.env.actions)))

function make_q_network(env, hidden_layer_size=32)
    QNetwork(env, Chain(
        Dense(length(env.state) + length(env.actions), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first))
end

Flux.@nograd function y(policy, sars; discount_factor=1)
    if sars.f
        sars.r
    else
        sars.r + discount_factor * maximum(policy.q(sars.s′, a′) for a′ in policy.env.actions)
    end
end

function q_loss(policy, sars)
    sum(sars) do sars
        (policy.q(sars.s, sars.a) - y(policy, sars))^2
    end / length(sars)
end

function train_policy!(policy, sars)
    append!(policy.replay_buffer, sars)
    q_optimizer = ADAM()
    for fit_iteration in 1:100
        Flux.train!(
            sars -> q_loss(policy, sars),
            Flux.params(policy.q),
            [(sample(policy.replay_buffer, 100),)],
            q_optimizer)
    end
end

mutable struct Policy <: AbstractPolicy
    env  # The environment is not part of the policy, but the policy depends on the environment.
    q
    action_noise
    replay_buffer
    train_policy!
end

function Reinforce.action(policy::Policy, r, s, A)
    actions = policy.env.actions
    actions[argmax([policy.q(s, a) + randn() * policy.action_noise for a in actions])]
end

function make_default_policy(env)
    Policy(env, make_q_network(env), 0.1, CircularBuffer(1_000_000), train_policy!)
end

end  # end module DQN

include("base.jl")

function run()
    env = GymEnv(:CartPole, :v1)
    policy = DQN.make_default_policy(env)
    run_until_reward(policy, 495, batch_size=10, adaptive_batch_size=false)
end
