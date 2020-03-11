using BSON
using Distributed
using Flux
using OpenAIGym
using Plots
using Printf
using ProgressMeter
using Sars
import Runner: train_policy!

pyplot()

const env = GymEnv(:LunarLander, :v2)

struct Policy{Π, Q, V} <: AbstractPolicy
    π :: Π
    q :: Q
    v :: V
end

Policy() = Policy(make_π_network(), make_q_network(), make_v_network())

Reinforce.action(policy::Policy, r, s, A) = sample(env.actions.items, Weights(policy.π(s)))

function make_π_network(hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        softmax)
end

struct QNetwork{T}
    network :: T
end
Flux.@treelike QNetwork
(q::QNetwork)(s, a) = q.network(vcat(s, Flux.onehot(a, env.actions.items)))

function make_q_network(hidden_layer_size=32)
    QNetwork(Chain(
        Dense(length(env.state) + length(env.actions), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first))
end

function make_v_network(hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first)
end

Flux.@nograd a_to_π_index(a) = indexin(a, env.actions.items)[1]

clip(n, ϵ) = clamp(n, 1 - ϵ, 1 + ϵ)

function π_loss(policy₀, policy′, sars, ϵ=0.2)
    -sum(sars) do sars
        π₀ = policy₀.π(sars.s)
        π′ = policy′.π(sars.s)
        a₀ = π₀[a_to_π_index(sars.a)]
        a′ = π′[a_to_π_index(sars.a)]
        advantage = policy₀.q(sars.s, sars.a) - policy₀.v(sars.s)
        a_ratio = a′ / a₀
        min(
            a_ratio * advantage,
            clip(a_ratio, ϵ) * advantage)
    end / length(sars)
end

function q_loss(policy, sars)
    sum(sars) do sars
        (policy.q(sars.s, sars.a) - sars.q)^2
    end / length(sars)
end

function v_loss(policy, sars)
    sum(sars) do sars
        (policy.v(sars.s) - sars.q)^2
    end / length(sars)
end

function train_policy!(policy::Policy, sars)
    fill_q!(sars, discount_factor=0.99)
    v_optimizer = ADAM()
    q_optimizer = ADAM()
    π_optimizer = ADAM()
    @showprogress "Fitting v: " for fit_iteration in 1:1000
        Flux.train!(sars -> v_loss(policy, sars), Flux.params(policy.v), [(sample(sars, 100),)], v_optimizer)
    end
    @showprogress "Fitting q: " for fit_iteration in 1:1000
        Flux.train!(sars -> q_loss(policy, sars), Flux.params(policy.q), [(sample(sars, 100),)], q_optimizer)
    end
    policy₀ = deepcopy(policy)
    policy′ = policy
    @showprogress "Fitting π: " for fit_iteration in 1:1000
        Flux.train!(sars -> π_loss(policy₀, policy′, sars), Flux.params(policy′.π), [(sample(sars, 100),)], π_optimizer)
    end
end
