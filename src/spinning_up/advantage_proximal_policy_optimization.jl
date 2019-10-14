module AdvantagePPO

using Flux
using OpenAIGym

struct SARS
    s
    a
    r
    q
    s′
    f
end

function fill_q(sars; discount_factor=1)
    sars′ = []
    q = 0
    for i in length(sars):-1:1
        q *= discount_factor
        if sars[i].f
            q = 0
        end
        q += sars[i].r
        push!(sars′, SARS(
            sars[i].s,
            sars[i].a,
            sars[i].r,
            q,
            sars[i].s′,
            sars[i].f))
    end
    sars′
end

function make_π_network(env, hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, length(env.actions), identity),
        softmax)
end

struct QNetwork
    env
    network
end
(q::QNetwork)(s, a) = q.network(vcat(s, Flux.onehot(a, q.env.actions)))
Flux.@treelike QNetwork

function make_q_network(env, hidden_layer_size=32)
    QNetwork(env, Chain(
        Dense(length(env.state) + length(env.actions), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first))
end

function make_v_network(env, hidden_layer_size=32)
    Chain(
        Dense(length(env.state), hidden_layer_size, swish),
        Dense(hidden_layer_size, hidden_layer_size, swish),
        Dense(hidden_layer_size, 1, identity),
        first)
end

a_to_π_index(env, a) = indexin(a, env.actions.items)[1]

clip(x, lo, hi) = clamp(x, Float32(lo), Float32(hi))

function π_loss(policy₀, policy′, sars, ϵ=0.1)
    -sum(sars) do sars
        π₀ = policy₀.π(sars.s)
        π′ = policy′.π(sars.s)
        a₀ = π₀[a_to_π_index(policy₀.env, sars.a)]
        a′ = π′[a_to_π_index(policy′.env, sars.a)]
        advantage = policy₀.q(sars.s, sars.a) - policy₀.v(sars.s)
        min((a′ / a₀) * advantage, clip(a′ / a₀, 1 - ϵ, 1 + ϵ) * advantage)
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

const π_optimizer = AMSGrad()
const q_optimizer = AMSGrad()
const v_optimizer = AMSGrad()

function train_policy!(policy, sars)
    sars = fill_q(sars)
    for fit_iteration in Iterators.countfrom(1)
        pre_loss = q_loss(policy, sars)
        for _ in 1:10
            Flux.train!(sars -> q_loss(policy, sars), Flux.params(policy.q), [(sample(sars, 100),)], q_optimizer)
        end
        post_loss = q_loss(policy, sars)
        if post_loss >= pre_loss || fit_iteration >= 10; break end
    end
    for fit_iteration in Iterators.countfrom(1)
        pre_loss = v_loss(policy, sars)
        for _ in 1:10
            Flux.train!(sars -> v_loss(policy, sars), Flux.params(policy.v), [(sample(sars, 100),)], v_optimizer)
        end
        post_loss = v_loss(policy, sars)
        if post_loss >= pre_loss || fit_iteration >= 10; break end
    end
    policy₀ = deepcopy(policy)
    policy′ = policy
    for fit_iteration in Iterators.countfrom(1)
        pre_loss = π_loss(policy₀, policy′, sars)
        for _ in 1:10
            Flux.train!(sars -> π_loss(policy₀, policy′, sars), Flux.params(policy′.π), [(sample(sars, 100),)], π_optimizer)
        end
        post_loss = π_loss(policy₀, policy′, sars)
        if post_loss >= pre_loss || fit_iteration >= 10; break end
    end
end

struct Policy <: AbstractPolicy
    env  # The environment is not part of the policy, but the policy depends on the environment.
    π
    q
    v
    train_policy!
end

Reinforce.action(policy::Policy, r, s, A) = sample(policy.env.actions, Weights(policy.π(s)))

function make_default_policy(env)
    Policy(env, make_π_network(env), make_q_network(env), make_v_network(env), train_policy!)
end

end  # end module AdvantagePPO

include("base.jl")

function run()
    env = GymEnv(:CartPole, :v1)
    policy = AdvantagePPO.make_default_policy(env)
    run_until_reward(policy, 495)
end
