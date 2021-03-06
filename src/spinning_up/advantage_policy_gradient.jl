module AdvantagePG

using Flux
using OpenAIGym

Flux.@nograd onehot(x, set::DiscreteSet) = Flux.onehot(x, set.items)

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
Flux.@treelike QNetwork
(q::QNetwork)(s, a) = q.network(vcat(s, onehot(a, q.env.actions)))

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

Flux.@nograd a_to_π_index(env, a) = indexin(a, env.actions.items)[1]

Flux.@nograd episode_count(sars) = length(filter(sars -> sars.f, sars))

function π_loss(policy, sars)
    -sum(sars) do sars
        Φ = policy.q(sars.s, sars.a) - policy.v(sars.s)
        log(policy.π(sars.s)[a_to_π_index(policy.env, sars.a)]) * Φ
    end / episode_count(sars)
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

const π_optimizer = ADAM()

function train_policy!(policy, sars)
    sars = fill_q(sars)
    v_optimizer = ADAM()
    q_optimizer = ADAM()
    for fit_iteration in 1:1000
        Flux.train!(sars -> v_loss(policy, sars), Flux.params(policy.v), [(sample(sars, 100),)], v_optimizer)
    end
    for fit_iteration in 1:1000
        Flux.train!(sars -> q_loss(policy, sars), Flux.params(policy.q), [(sample(sars, 100),)], q_optimizer)
    end
    Flux.train!(sars -> π_loss(policy, sars), Flux.params(policy.π), [(sars,)], π_optimizer)
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

end  # end module AdvantagePG

include("base.jl")

function run()
    env = GymEnv(:CartPole, :v1)
    policy = AdvantagePG.make_default_policy(env)
    run_until_reward(policy, 495)
end
