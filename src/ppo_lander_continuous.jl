using Flux
using OpenAIGym
using ProgressMeter
using Sars
import Runner

const env = GymEnv(:LunarLanderContinuous, :v2)

struct Policy{Π, Q, V} <: AbstractPolicy
    π :: Π
    σ :: Vector{Float32}
    q :: Q
    v :: V
end

Policy() = Policy(make_π_network(), [0f0, 0f0], make_q_network(), make_v_network())

Reinforce.action(policy::Policy, r, s, A) = sample_π(policy, s)

function make_π_network(hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        x -> tanh.(x)*2)
end

struct QNetwork{T}
    network :: T
end
Flux.@treelike QNetwork
(q::QNetwork)(s, a) = q.network(vcat(s, a))

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

function pa(policy, sars)
    μ = policy.π(sars.s)
    σ = exp.(policy.σ)
    a = sars.a
    Σ = @. $sum((a-μ)^2 / σ^2 + 2*log(σ))
    exp((-1/2) * (Σ + length(a) * log(2*pi)))
end

function sample_π(policy, s)
    μ = policy.π(s)
    σ = exp.(policy.σ)
    a = μ + σ .* randn(length(μ))
    (n -> clamp(n, -1, 1)).(a)
end

clip(n, ϵ) = clamp(n, 1 - ϵ, 1 + ϵ)

function π_loss(policy₀, policy′, sars, ϵ=0.2)
    -sum(sars) do sars
        a₀ = pa(policy₀, sars)
        a′ = pa(policy′, sars)
        advantage = policy₀.q(sars.s, sars.a) - policy₀.v(sars.s)
        a_ratio = a′ / a₀
        r = min(
            a_ratio * advantage,
            clip(a_ratio, ϵ) * advantage)
        isnan(r) ? 0 : r
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

function Runner.train_policy!(policy::Policy, sars)
    fill_q!(sars, discount_factor=0.99)
    v_optimizer = ADAM()
    q_optimizer = ADAM()
    π_optimizer = ADAM(0.000_1)
    @showprogress "Fitting v: " for fit_iteration in 1:1000
        Flux.train!(sars -> v_loss(policy, sars), Flux.params(policy.v), [(sample(sars, 100),)], v_optimizer)
    end
    @showprogress "Fitting q: " for fit_iteration in 1:1000
        Flux.train!(sars -> q_loss(policy, sars), Flux.params(policy.q), [(sample(sars, 100),)], q_optimizer)
    end
    policy₀ = deepcopy(policy)
    policy′ = policy
    @showprogress "Fitting π: " for fit_iteration in 1:1000
        Flux.train!(
            sars -> π_loss(policy₀, policy′, sars),
            Flux.params(policy′.π, policy′.σ),
            [(sample(sars, 100),)],
            π_optimizer)
    end
end

Runner.environment(policy::Policy) = env
Runner.statetype(policy::Policy) = Vector{Float32}
Runner.actiontype(policy::Policy) = Vector{Float32}
