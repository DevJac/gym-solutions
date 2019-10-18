module SimplePPO

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

a_to_π_index(env, a) = indexin(a, env.actions.items)[1]

clip(x, lo, hi) = clamp(x, Float32(lo), Float32(hi))

function π_loss(policy₀, policy′, sars, ϵ=0.2)
    baseline = mean(sars.q for sars in sars)
    -sum(sars) do sars
        π₀ = policy₀.π(sars.s)
        π′ = policy′.π(sars.s)
        a₀ = π₀[a_to_π_index(policy₀.env, sars.a)]
        a′ = π′[a_to_π_index(policy′.env, sars.a)]
        min((a′ / a₀) * (sars.q - baseline), clip(a′ / a₀, 1 - ϵ, 1 + ϵ) * (sars.q - baseline))
    end / length(sars)
end

function train_policy!(policy, sars)
    sars = fill_q(sars)
    policy₀ = deepcopy(policy)
    policy′ = policy
    π_optimizer = ADAM()
    for fit_iteration in 1:1000
        Flux.train!(sars -> π_loss(policy₀, policy′, sars), Flux.params(policy′.π), [(sample(sars, 100),)], π_optimizer)
    end
end

struct Policy <: AbstractPolicy
    env  # The environment is not part of the policy, but the policy depends on the environment.
    π
    train_policy!
end

Reinforce.action(policy::Policy, r, s, A) = sample(policy.env.actions, Weights(policy.π(s)))

function make_default_policy(env)
    Policy(env, make_π_network(env), train_policy!)
end

end  # end module SimplePPO

include("base.jl")

function run()
    env = GymEnv(:CartPole, :v1)
    policy = SimplePPO.make_default_policy(env)
    run_until_reward(policy, 495)
end
